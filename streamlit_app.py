import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# ===== DATA CLEANING FUNCTIONS (MODIFIED FOR DONOR STANDARDIZATION) =====
def clean_and_process_data(df):
    """
    Clean and process student assessment data
    
    Parameters:
    df (pd.DataFrame): Raw dataframe from Excel
    
    Returns:
    pd.DataFrame: Cleaned and processed dataframe
    """
    
    initial_count = len(df)
    
    # ===== STEP 1: DATA CLEANING =====
    # Define pre and post question columns
    pre_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    post_questions = ['Q1_Post', 'Q2_Post', 'Q3_Post', 'Q4_Post', 'Q5_Post']
    
    # Condition 1: Remove rows where one set has values and the other is all NULL
    # If ANY pre question has values but ALL post questions are NULL
    has_any_pre = df[pre_questions].notna().any(axis=1)
    all_post_null = df[post_questions].isna().all(axis=1)
    remove_condition_1 = has_any_pre & all_post_null
    
    # If ALL pre questions are NULL but ANY post question has values
    all_pre_null = df[pre_questions].isna().all(axis=1)
    has_any_post = df[post_questions].notna().any(axis=1)
    remove_condition_2 = all_pre_null & has_any_post
    
    # Condition 3: Remove rows where BOTH pre and post are all NULL
    remove_condition_3 = all_pre_null & all_post_null
    
    # Remove rows that meet any of the conditions
    df = df[~(remove_condition_1 | remove_condition_2 | remove_condition_3)]
    
    cleaned_count = len(df)
    
    # ===== STEP 2: CALCULATE SCORES =====
    # Define answer columns
    pre_answers = ['Q1 Answer', 'Q2 Answer', 'Q3 Answer', 'Q4 Answer', 'Q5 Answer']
    post_answers = ['Q1_Answer_Post', 'Q2_Answer_Post', 'Q3_Answer_Post', 'Q4_Answer_Post', 'Q5_Answer_Post']
    
    # Calculate Pre-session scores
    df['Pre_Score'] = 0
    for q, ans in zip(pre_questions, pre_answers):
        df['Pre_Score'] += (df[q] == df[ans]).astype(int)
    
    # Calculate Post-session scores
    df['Post_Score'] = 0
    for q, ans in zip(post_questions, post_answers):
        df['Post_Score'] += (df[q] == df[ans]).astype(int)
    
    # ===== STEP 3: STANDARDIZE PROGRAM TYPES =====
    # Create a mapping for program types
    # SCB, SCC, SCM, SCP are combined into PCMB
    program_type_mapping = {
        'SCB': 'PCMB',
        'SCC': 'PCMB',
        'SCM': 'PCMB',
        'SCP': 'PCMB',
        'E-LOB': 'ELOB',
        'DLC-2': 'DLC',
        'DLC2': 'DLC'
    }
    
    # Apply the mapping
    df['Program Type'] = df['Program Type'].replace(program_type_mapping)
    
    # ===== STEP 4: CREATE PARENT CLASS =====
    # Extract parent class from Class column (e.g., "6-A" -> "6", "7-B" -> "7")
    df['Parent_Class'] = df['Class'].astype(str).str.extract(r'^(\d+)')[0]
    
    # Filter for grades 6-10 only
    df = df[df['Parent_Class'].isin(['6', '7', '8', '9', '10'])]

    # ===== STEP 5: STANDARDIZE DONOR NAMES (NEW FIX FOR CASE-SENSITIVITY) =====
    if 'Donor' in df.columns:
        # Convert all donor names to uppercase to treat 'Adobe', 'ADOBE', 'adobe' as a single entity.
        df['Donor'] = df['Donor'].astype(str).str.upper()
    # -------------------------------------------------------------------------
    
    # ===== STEP 6: STANDARDIZE SUBJECT NAMES (NEW FIX FOR CASE-SENSITIVITY AND VARIATIONS) =====
    if 'Subject' in df.columns:
        # Convert Subject column to string and then uppercase for initial standardization
        df['Subject'] = df['Subject'].astype(str).str.upper()
        
        # Apply explicit mapping to handle variations and consolidate
        df['Subject'] = df['Subject'].replace({
            'SCIENCE': 'SCIENCE',
            'MATH': 'MATH',
            'SCIENCE ': 'SCIENCE', # Catch trailing spaces
            'MATH ': 'MATH'
            # This step is mainly redundant after .str.upper() but ensures explicit consolidation.
        }, regex=False)
        
    # ------------------------------------------------------------------------------------------
    
    return df, initial_count, cleaned_count

# ===== TAB 8: SUBJECT ANALYSIS (UNCHANGED) =====
def tab8_subject_analysis(df):
    """
    Generates the Subject-wise Performance and Participation Analysis, 
    including breakdowns by Region.
    """
    st.header("Subject-wise Performance Analysis")
    st.markdown("### Performance, Participation, and Assessment Count by Subject")
    
    # Check for required columns
    required_cols = ['Subject', 'Region']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Columns {required_cols} not found in the data. Cannot perform Subject-Region Analysis.")
        return

    # Calculate Subject Statistics (Overall)
    subject_stats = df.groupby('Subject').agg(
        Num_Students=('Student Id', 'nunique'),        # Number of unique students
        Num_Assessments=('Student Id', 'count'),       # Total number of assessments (rows)
        Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
        Avg_Post_Score_Raw=('Post_Score', 'mean')
    ).reset_index()

    # Calculate percentages and improvement
    subject_stats['Avg Pre Score %'] = (subject_stats['Avg_Pre_Score_Raw'] / 5) * 100
    subject_stats['Avg Post Score %'] = (subject_stats['Avg_Post_Score_Raw'] / 5) * 100
    subject_stats['Improvement %'] = subject_stats['Avg Post Score %'] - subject_stats['Avg Pre Score %']
    
    # Sort in ascending order of PRE Score %
    subject_stats = subject_stats.sort_values('Avg Pre Score %', ascending=True)
    
    # --- Visualization: Performance (Overall) ---
    st.subheader("📈 Overall Subject Performance Comparison (Pre vs. Post)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=subject_stats['Subject'],
        y=subject_stats['Avg Pre Score %'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_stats['Avg Pre Score %']],
        textposition='top center',
        textfont=dict(size=12, color='#3498db')
    ))
    
    fig.add_trace(go.Scatter(
        x=subject_stats['Subject'],
        y=subject_stats['Avg Post Score %'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_stats['Avg Post Score %']],
        textposition='top center',
        textfont=dict(size=12, color='#e74c3c')
    ))
    
    fig.update_layout(
        title='Subject-wise Pre and Post Assessment Scores (Ascending by Pre Score)',
        xaxis_title='Subject',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(type='category', gridcolor='#404040') # Force category type
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate Subject-Region Statistics for breakdowns
    subject_region_stats = df.groupby(['Subject', 'Region']).agg(
        Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
        Avg_Post_Score_Raw=('Post_Score', 'mean')
    ).reset_index()
    
    subject_region_stats['Pre_Score_Pct'] = (subject_region_stats['Avg_Pre_Score_Raw'] / 5) * 100
    subject_region_stats['Post_Score_Pct'] = (subject_region_stats['Avg_Post_Score_Raw'] / 5) * 100
    
    st.markdown("---")
    
    # ====================================================================
    # 1. NEW GRAPH: Subject Analysis per Region 
    # ====================================================================
    st.subheader("🗺️ Subject Performance within a Selected Region")
    
    # CHANGED TO MULTI-SELECT
    unique_regions = sorted(df['Region'].unique())
    selected_regions_for_subject = st.multiselect("Select Region(s) for Subject Breakdown", unique_regions, default=unique_regions, key='region_subject_select')

    # Filter data for the selected region(s)
    region_subject_data = subject_region_stats[subject_region_stats['Region'].isin(selected_regions_for_subject)].copy()
    
    # Create the multi-trace figure
    fig_subj_region = go.Figure()
    color_scale = px.colors.qualitative.Plotly 
    color_index = 0

    for region in selected_regions_for_subject:
        # Filter for the specific region's data
        single_region_data = region_subject_data[region_subject_data['Region'] == region].copy()
        
        # Sort in ascending order of PRE Score % for consistent axis
        single_region_data = single_region_data.sort_values('Pre_Score_Pct', ascending=True)

        base_color = color_scale[color_index % len(color_scale)]

        # Pre-Session
        fig_subj_region.add_trace(go.Scatter(
            x=single_region_data['Subject'],
            y=single_region_data['Pre_Score_Pct'],
            mode='lines+markers',
            name=f'{region} (Pre)',
            line=dict(color=base_color, width=3, dash='dot'),
            marker=dict(size=10, symbol='circle'),
            hovertemplate = f'<b>Region:</b> {region}<br><b>Subject:</b> %{{x}}<br><b>Pre Score:</b> %{{y:.1f}}%<extra></extra>'
        ))
        
        # Post-Session
        fig_subj_region.add_trace(go.Scatter(
            x=single_region_data['Subject'],
            y=single_region_data['Post_Score_Pct'],
            mode='lines+markers',
            name=f'{region} (Post)',
            line=dict(color=base_color, width=3, dash='solid'),
            marker=dict(size=10, symbol='square'),
            hovertemplate = f'<b>Region:</b> {region}<br><b>Subject:</b> %{{x}}<br><b>Post Score:</b> %{{y:.1f}}%<extra></extra>'
        ))
        
        color_index += 1

    # Title adjustment for multi-select
    title_regions = ", ".join(selected_regions_for_subject) if len(selected_regions_for_subject) <= 3 else f"{len(selected_regions_for_subject)} Regions"

    fig_subj_region.update_layout(
        title=f'Subject Performance Comparison for **{title_regions}**',
        xaxis_title='Subject',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(type='category', gridcolor='#404040') # Force category type
    )
    
    st.plotly_chart(fig_subj_region, use_container_width=True)
    
    st.markdown("---")

    # ====================================================================
    # 2. NEW GRAPH: Region Analysis per Subject 
    # ====================================================================
    st.subheader("📍 Region Performance for a Selected Subject")
    
    # CHANGED TO MULTI-SELECT
    unique_subjects = sorted(df['Subject'].unique())
    selected_subjects_for_region = st.multiselect("Select Subject(s) for Region Breakdown", unique_subjects, default=unique_subjects, key='subject_region_select')

    # Filter data for the selected subject(s)
    subject_region_data_multi = subject_region_stats[subject_region_stats['Subject'].isin(selected_subjects_for_region)].copy()
    
    # Create the multi-trace figure
    fig_region_subj = go.Figure()
    color_scale = px.colors.qualitative.Plotly 
    color_index = 0

    for subject in selected_subjects_for_region:
        # Filter for the specific subject's data
        single_subject_data = subject_region_data_multi[subject_region_data_multi['Subject'] == subject].copy()
        
        # Sort in ascending order of PRE Score %
        single_subject_data = single_subject_data.sort_values('Region', ascending=True)

        base_color = color_scale[color_index % len(color_scale)]
        
        # Pre-Session
        fig_region_subj.add_trace(go.Scatter(
            x=single_subject_data['Region'],
            y=single_subject_data['Pre_Score_Pct'],
            mode='lines+markers',
            name=f'{subject} (Pre)',
            line=dict(color=base_color, width=3, dash='dot'),
            marker=dict(size=10, symbol='circle'),
            hovertemplate = f'<b>Subject:</b> {subject}<br><b>Region:</b> %{{x}}<br><b>Pre Score:</b> %{{y:.1f}}%<extra></extra>'
        ))
        
        # Post-Session
        fig_region_subj.add_trace(go.Scatter(
            x=single_subject_data['Region'],
            y=single_subject_data['Post_Score_Pct'],
            mode='lines+markers',
            name=f'{subject} (Post)',
            line=dict(color=base_color, width=3, dash='solid'),
            marker=dict(size=10, symbol='square'),
            hovertemplate = f'<b>Subject:</b> {subject}<br><b>Region:</b> %{{x}}<br><b>Post Score:</b> %{{y:.1f}}%<extra></extra>'
        ))
        
        color_index += 1
    
    # Title adjustment for multi-select
    title_subjects = ", ".join(selected_subjects_for_region) if len(selected_subjects_for_region) <= 3 else f"{len(selected_subjects_for_region)} Subjects"

    fig_region_subj.update_layout(
        title=f'Region Performance Comparison for **{title_subjects}**',
        xaxis_title='Region',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(type='category', gridcolor='#404040') # Force category type
    )
    
    st.plotly_chart(fig_region_subj, use_container_width=True)
    
    # --- Detailed Table and Participation Metrics (UNCHANGED) ---
    st.markdown("---")
    st.subheader("📋 Subject Participation and Detailed Metrics (Overall)")

    # Create the display dataframe
    display_subject_stats = subject_stats.copy()
    display_subject_stats = display_subject_stats[[
        'Subject', 
        'Num_Students', 
        'Num_Assessments', 
        'Avg Pre Score %', 
        'Avg Post Score %', 
        'Improvement %'
    ]]
    
    display_subject_stats.columns = [
        'Subject', 
        'Unique Students', 
        'Total Assessments', 
        'Avg Pre %', 
        'Avg Post %', 
        'Improvement %'
    ]
    
    # Apply string formatting
    display_subject_stats['Avg Pre %'] = display_subject_stats['Avg Pre %'].apply(lambda x: f"{x:.1f}%")
    display_subject_stats['Avg Post %'] = display_subject_stats['Avg Post %'].apply(lambda x: f"{x:.1f}%")
    display_subject_stats['Improvement %'] = display_subject_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_subject_stats, hide_index=True, use_container_width=True)
    
    # --- Key Participation Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Subjects", len(subject_stats))
    with col2:
        st.metric("Total Unique Students Assessed", subject_stats['Num_Students'].sum())
    with col3:
        st.metric("Total Assessments Conducted", subject_stats['Num_Assessments'].sum())
        
    # Download Button
    st.markdown("---")
    # Must use the float columns for download, not the display string columns
    subject_csv_for_download = subject_stats[[
        'Subject', 
        'Num_Students', 
        'Num_Assessments', 
        'Avg Pre Score %', 
        'Avg Post Score %', 
        'Improvement %'
    ]].copy()
    subject_csv_for_download.columns = [
        'Subject', 
        'Unique Students', 
        'Total Assessments', 
        'Avg Pre %', 
        'Avg Post %', 
        'Improvement %'
    ]
    subject_csv = subject_csv_for_download.to_csv(index=False)
    st.download_button(
        "📥 Download Subject Analysis Data (CSV)",
        subject_csv,
        "subject_analysis.csv",
        "text/csv"
    )
    
    return subject_stats # Return for use in the main download section

# ===== MAIN APPLICATION (MODIFIED FILTERS) =====

# Title and description
st.title("📊 Student Assessment Analysis Platform")
st.markdown("### Upload, Clean, and Analyze Student Performance Data")

# File uploader
uploaded_file = st.file_uploader("Upload Student Data Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    
    # Load and clean data
    with st.spinner("Loading and cleaning data..."):
        try:
            raw_df = pd.read_excel(uploaded_file)
            
            # Basic checks for required columns
            required_check_cols = ['Date_Post', 'Donor', 'Subject', 'Region', 'Student Id', 'Class', 'Program Type', 'Q1', 'Q1_Post']
            missing_cols = [col for col in required_check_cols if col not in raw_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}. Please add these columns and try again.")
                st.stop()
                
            df, initial_count, cleaned_count = clean_and_process_data(raw_df)
            
            # Show cleaning summary
            st.success("✅ Data cleaned successfully!")
            
            # Calculate Wastage Rate
            records_removed = initial_count - cleaned_count
            if initial_count > 0:
                wastage_rate = (records_removed / initial_count) * 100
                wastage_rate_str = f"{wastage_rate:.1f}%"
            else:
                wastage_rate = 0
                wastage_rate_str = "0.0%"
            
            # Update the column layout to fit the new metric
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Records", initial_count)
            with col2:
                st.metric("Records Removed", records_removed)
            with col3:
                st.metric("Final Records", cleaned_count)
            with col4:
                st.metric("Wastage Rate", wastage_rate_str, delta=f"{wastage_rate:.1f}% loss") # Use delta to highlight it as a loss rate
            
            # Option to download cleaned data
            st.markdown("---")
            cleaned_excel = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Data (CSV)",
                data=cleaned_excel,
                file_name="cleaned_student_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()
    
    # Sidebar filters (MODIFIED TO MULTI-SELECT)
    st.sidebar.header("🔍 Filters (Multi-Select Enabled)")
    
    # WARNING REMOVED as multi-select handles 'All' by default selection

    # Region filter
    all_regions = sorted(df['Region'].unique().tolist())
    # Defaulting to all regions enables full analysis unless manually filtered
    selected_regions = st.sidebar.multiselect("Select Region(s)", all_regions, default=all_regions)
    
    # Program Type filter
    all_programs = sorted(df['Program Type'].unique().tolist())
    selected_programs = st.sidebar.multiselect("Select Program Type(s)", all_programs, default=all_programs)
    
    # Parent Class filter
    all_classes = sorted(df['Parent_Class'].unique().tolist())
    selected_classes = st.sidebar.multiselect("Select Grade(s)", all_classes, default=all_classes)
    
    # Apply filters (MODIFIED TO .isin() LOGIC)
    filtered_df = df.copy()
    
    # Use .isin() for multi-selection
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
    if selected_programs:
        filtered_df = filtered_df[filtered_df['Program Type'].isin(selected_programs)]
    if selected_classes:
        filtered_df = filtered_df[filtered_df['Parent_Class'].isin(selected_classes)]
    
    # ===== KEY METRICS (UNCHANGED) =====
    st.markdown("---")
    st.subheader("📊 Key Performance Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        avg_pre = (filtered_df['Pre_Score'].mean() / 5) * 100
        st.metric("Avg Pre Score", f"{avg_pre:.1f}%")
    
    with col2:
        avg_post = (filtered_df['Post_Score'].mean() / 5) * 100
        st.metric("Avg Post Score", f"{avg_post:.1f}%")
    
    with col3:
        improvement = avg_post - avg_pre
        st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
    
    
    if not filtered_df.empty:
        # 1. Identify unique tests per student
        # A unique test is defined by Student, Content, Class, School AND Date.
        unique_student_tests = filtered_df.groupby(
            ['Student Id', 'Content Id', 'Class', 'School Name', 'Date_Post']
        ).size().reset_index(name='count')
        
        # 2. Count how many such unique tests each student has taken
        student_activity = unique_student_tests.groupby('Student Id').size().reset_index(name='Visible_Tests')

        # 3. Calculate metrics
        avg_tests = student_activity['Visible_Tests'].mean()
        max_tests = student_activity['Visible_Tests'].max()
        min_tests = student_activity['Visible_Tests'].min()
        
        # 4. Calculate test frequency distribution for col 5 display
        test_counts = student_activity.groupby('Visible_Tests').size().reset_index(name='Num Students')
        test_counts.columns = ['Tests Taken', 'Num Students']
        test_counts['Percentage'] = (test_counts['Num Students'] / test_counts['Num Students'].sum()) * 100
        test_counts = test_counts.sort_values('Tests Taken', ascending=False)
    else:
        avg_tests, max_tests, min_tests = 0, 0, 0
        test_counts = pd.DataFrame()
    
    with col4:
        st.metric("Avg Tests/Student", f"{avg_tests:.1f}")

    with col5:
        st.metric("Max Tests/Student", f"{max_tests}")
        
        # Display frequency distribution (only if data exists)
        if not test_counts.empty:
            st.markdown("###### Test Frequency Distribution")
            
            # Display only the top 10 frequencies or fewer if less exist
            for index, row in test_counts.head(min(10, len(test_counts))).iterrows():
                # Cast to int to prevent .0 from showing
                st.markdown(f"* **{int(row['Tests Taken'])}x**: {int(row['Num Students'])} Students ({row['Percentage']:.1f}%)")


    with col6:
        st.metric("Min Tests/Student", f"{min_tests}")
    
    # ===== TABS FOR DIFFERENT ANALYSES (UNCHANGED) =====
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📍 Region Analysis", "👤 Instructor Analysis", "📚 Grade Analysis", "📊 Program Type Analysis", "👥 Student Participation", "🏫 School Analysis", "💰 Donor Analysis", "🔬 Subject Analysis"])
    
    # Placeholder for subject_stats for download section
    subject_stats = None

    # ===== TAB 1: REGION ANALYSIS (MODIFIED PROGRAM TYPE ANALYSIS) =====
    with tab1:
        st.header("Region-wise Performance Analysis")
        
        # Overall region analysis
        region_stats = filtered_df.groupby('Region').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        region_stats['Pre_Score_Pct'] = (region_stats['Pre_Score'] / 5) * 100
        region_stats['Post_Score_Pct'] = (region_stats['Post_Score'] / 5) * 100
        region_stats['Improvement'] = region_stats['Post_Score_Pct'] - region_stats['Pre_Score_Pct']
        
        # Sort in ascending order of PRE Score Pct
        region_stats = region_stats.sort_values('Pre_Score_Pct', ascending=True)
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=12, color='#2ecc71')
        ))
        
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=12, color='#e67e22')
        ))
        
        fig.update_layout(
            title='Region-wise Performance Comparison (Ascending by Pre Score)',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            hovermode='x unified',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040') # Force category type
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Region by Program Type (Multi-Select Version)
        st.subheader("Region Analysis by Program Type (Multi-Select)")
        
        program_region_stats = filtered_df.groupby(['Region', 'Program Type']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        # Change selectbox to multiselect
        unique_programs = sorted(filtered_df['Program Type'].unique())
        selected_program_types = st.multiselect("Select Program Type(s) for Detailed View", 
                                             unique_programs, default=unique_programs, key='tab1_program_select')
        
        fig2 = go.Figure()
        
        # Define a list of distinct colors for the lines
        color_scale = px.colors.qualitative.Plotly 
        color_index = 0

        # Iterate over selected programs and add traces for each
        for program in selected_program_types:
            prog_data = program_region_stats[program_region_stats['Program Type'] == program]
            # Sort by Region Name for consistent X-axis order when comparing multiple programs
            prog_data = prog_data.sort_values('Region', ascending=True) 

            # Get the base color
            base_color = color_scale[color_index % len(color_scale)]
            
            # Pre-Session Line (dotted)
            fig2.add_trace(go.Scatter(
                x=prog_data['Region'],
                y=prog_data['Pre_Score_Pct'],
                mode='lines+markers',
                name=f'{program} (Pre)',
                line=dict(color=base_color, width=3, dash='dot'), 
                marker=dict(size=10, symbol='circle'),
                hovertemplate = '<b>Region:</b> %{x}<br><b>Pre Score:</b> %{y:.1f}%<br>'
            ))
            
            # Post-Session Line (solid)
            fig2.add_trace(go.Scatter(
                x=prog_data['Region'],
                y=prog_data['Post_Score_Pct'],
                mode='lines+markers',
                name=f'{program} (Post)',
                line=dict(color=base_color, width=3, dash='solid'), 
                marker=dict(size=10, symbol='square'),
                hovertemplate = '<b>Region:</b> %{x}<br><b>Post Score:</b> %{y:.1f}%<br>'
            ))
            
            color_index += 1 # Move to the next color

        # Update Layout
        if selected_program_types:
            title_programs = ", ".join(selected_program_types) if len(selected_program_types) <= 3 else f"{len(selected_program_types)} Programs"
        else:
            title_programs = "No Programs Selected"
            
        fig2.update_layout(
            title=f'Region-wise Performance Comparison for {title_programs}',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Top performing and most improved regions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Scoring Regions (Post-Session)")
            top_scoring = region_stats.nlargest(5, 'Post_Score_Pct')[['Region', 'Post_Score_Pct']]
            top_scoring['Post_Score_Pct'] = top_scoring['Post_Score_Pct'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_scoring, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("📈 Most Improved Regions (Adaptation)")
            most_improved = region_stats.nlargest(5, 'Improvement')[['Region', 'Improvement']]
            most_improved['Improvement'] = most_improved['Improvement'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(most_improved, hide_index=True, use_container_width=True)
    
    # ===== TAB 2: INSTRUCTOR ANALYSIS (UNCHANGED) =====
    with tab2:
        st.header("Instructor-wise Performance Analysis")
        
        instructor_stats = filtered_df.groupby('Instructor Name').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        instructor_stats['Pre_Score_Pct'] = (instructor_stats['Pre_Score'] / 5) * 100
        instructor_stats['Post_Score_Pct'] = (instructor_stats['Post_Score'] / 5) * 100
        instructor_stats['Improvement'] = instructor_stats['Post_Score_Pct'] - instructor_stats['Pre_Score_Pct']
        
        # The general stats used for tables are sorted descending
        instructor_stats_for_table = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
        
        # Show top N instructors
        top_n = st.slider("Number of instructors to display", 5, 20, 10)
        
        # Get top N performers, then sort them ascending by PRE for the plot
        top_instructors = instructor_stats_for_table.nlargest(top_n, 'Post_Score_Pct').sort_values('Pre_Score_Pct', ascending=True)

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in top_instructors['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig.add_trace(go.Scatter(
            x=top_instructors['Instructor Name'],
            y=top_instructors['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in top_instructors['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Instructors by Post-Session Performance (Ascending by Pre Score)',
            xaxis_title='Instructor',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(tickangle=-45, gridcolor='#404040') # Category is default for names
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Instructor rankings (using the descending table stats)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Instructors")
            top_perf = instructor_stats_for_table.nlargest(10, 'Post_Score_Pct')[['Instructor Name', 'Post_Score_Pct', 'Student Id']]
            top_perf.columns = ['Instructor', 'Post Score %', 'Students']
            top_perf['Post Score %'] = top_perf['Post Score %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_perf, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("📈 Best Adaptation (Improvement)")
            best_adapt = instructor_stats_for_table.nlargest(10, 'Improvement')[['Instructor Name', 'Improvement', 'Student Id']]
            best_adapt.columns = ['Instructor', 'Improvement %', 'Students']
            best_adapt['Improvement %'] = best_adapt['Improvement %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(best_adapt, hide_index=True, use_container_width=True)
        
        # All Instructors Assessment Count
        st.markdown("---")
        st.subheader("📋 Complete Instructor List - Assessment Count")
        
        # FIX APPLIED: Correctly calculate assessments using Date_Post
        filtered_df['Assessment_Session_Key'] = (
            filtered_df['Content Id'].astype(str) + '_' + 
            filtered_df['Class'].astype(str) + '_' + 
            filtered_df['School Name'].fillna('NA').astype(str) + '_' +
            filtered_df['Date_Post'].astype(str) # Added Date_Post
        )
        
        # FIX APPLIED: Group primarily by Instructor Name and use the mode for Login Id/Region
        # This prevents duplicate instructor entries if there is a single character difference in Login ID
        all_instructors = filtered_df.groupby(['Instructor Name']).agg(
            Instructor_Login_Id=('Instructor Login Id', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
            Assessment_Session_Key=('Assessment_Session_Key', 'nunique'), # This correctly counts distinct sessions
            Student_Id=('Student Id', 'count'),
            Region=('Region', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]) # Most common region
        ).reset_index()

        all_instructors.columns = ['Instructor Name', 'Login ID', 'Total Sessions', 'Total Assessments', 'Region']
        all_instructors = all_instructors.sort_values('Total Assessments', ascending=False)
        
        st.dataframe(all_instructors, hide_index=True, use_container_width=True)
        
        # Total metrics
        total_unique_instructors = filtered_df['Instructor Name'].nunique()
        total_sessions = all_instructors['Total Sessions'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unique Instructors", total_unique_instructors)
        with col2:
            st.metric("Total Unique Assessment Sessions", total_sessions)
        with col3:
            st.metric("Total Assessments Conducted", all_instructors['Total Assessments'].sum())

        # Download button
        instructor_csv = all_instructors.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Instructor Assessment List",
            instructor_csv,
            "instructor_assessments.csv",
            "text/csv"
        )
        
        # Instructors per Region
        st.markdown("---")
        st.subheader("👥 Number of Instructors per Region")
        instructors_per_region = filtered_df.groupby('Region')['Instructor Name'].nunique().reset_index()
        instructors_per_region.columns = ['Region', 'Number of Instructors']
        instructors_per_region = instructors_per_region.sort_values('Number of Instructors', ascending=False)
        
        # Create bar chart
        fig_inst_region = go.Figure()
        fig_inst_region.add_trace(go.Bar(
            x=instructors_per_region['Region'],
            y=instructors_per_region['Number of Instructors'],
            marker_color='#3498db',
            text=instructors_per_region['Number of Instructors'],
            textposition='outside',
            textfont=dict(size=14)
        ))
        
        fig_inst_region.update_layout(
            title='Number of Instructors by Region',
            xaxis_title='Region',
            yaxis_title='Number of Instructors',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        
        st.plotly_chart(fig_inst_region, use_container_width=True)
        
        # Display table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(instructors_per_region, hide_index=True, use_container_width=True)
        with col2:
            # FIX: Use the explicit unique count
            st.metric("Total Unique Instructors", total_unique_instructors)
            st.metric("Average per Region", f"{instructors_per_region['Number of Instructors'].mean():.1f}")

    # ===== TAB 3: GRADE ANALYSIS (UNCHANGED) =====
    with tab3:
        st.header("Grade-wise Performance Analysis")
        grade_stats = filtered_df.groupby('Parent_Class').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        grade_stats['Pre_Score_Pct'] = (grade_stats['Pre_Score'] / 5) * 100
        grade_stats['Post_Score_Pct'] = (grade_stats['Post_Score'] / 5) * 100
        grade_stats['Improvement'] = grade_stats['Post_Score_Pct'] - grade_stats['Pre_Score_Pct']
        
        # Sort in ascending order of PRE Score Pct
        grade_stats = grade_stats.sort_values('Pre_Score_Pct', ascending=True)

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#1abc9c', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#1abc9c')
        ))
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f1c40f', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#f1c40f')
        ))
        
        fig.update_layout(
            title='Grade-wise Performance Comparison (Ascending by Pre Score)',
            xaxis_title='Grade',
            yaxis_title='Average Score (%)',
            hovermode='x unified',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040') # Force category type
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Grade-wise participation (Bar Chart)
        st.markdown("---")
        st.subheader("Student Participation by Grade")
        
        # Ensure grade_stats is sorted by grade number for the bar chart
        grade_stats_participation = grade_stats.sort_values('Parent_Class')
        
        fig_part = go.Figure()
        fig_part.add_trace(go.Bar(
            x=grade_stats_participation['Parent_Class'],
            y=grade_stats_participation['Student Id'],
            marker_color='#3498db',
            text=grade_stats_participation['Student Id'],
            textposition='outside'
        ))
        
        fig_part.update_layout(
            title='Total Assessments Conducted by Grade',
            xaxis_title='Grade',
            yaxis_title='Total Assessments',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040')
        )
        st.plotly_chart(fig_part, use_container_width=True)
        
        # Detailed table
        st.markdown("---")
        st.subheader("📋 Detailed Grade Metrics")
        display_grade_stats = grade_stats.copy()
        display_grade_stats.columns = ['Grade', 'Avg Pre Score', 'Avg Post Score', 'Total Assessments', 'Pre %', 'Post %', 'Improvement %']
        
        # Format columns for display
        display_grade_stats['Pre %'] = display_grade_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_grade_stats['Post %'] = display_grade_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_grade_stats['Improvement %'] = display_grade_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        # Select and reorder columns for user display
        display_grade_stats = display_grade_stats[['Grade', 'Total Assessments', 'Pre %', 'Post %', 'Improvement %']]
        st.dataframe(display_grade_stats, hide_index=True, use_container_width=True)

    # ===== TAB 4: PROGRAM TYPE ANALYSIS (MODIFIED REGION ANALYSIS) =====
    with tab4:
        st.header("Program Type Performance Analysis")
        
        # 1. Overall Program Type Performance (Existing)
        program_stats = filtered_df.groupby('Program Type').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
        program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
        program_stats['Improvement'] = program_stats['Post_Score_Pct'] - program_stats['Pre_Score_Pct']
        
        # MODIFIED: Sort in ascending order of PRE Score Pct
        program_stats = program_stats.sort_values('Pre_Score_Pct', ascending=True)

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Pre_Score_Pct'],
            name='Pre-Session',
            marker_color='#3498db',
            text=[f"{val:.0f}%" for val in program_stats['Pre_Score_Pct']],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=program_stats['Program Type'],
            y=program_stats['Post_Score_Pct'],
            name='Post-Session',
            marker_color='#e74c3c',
            text=[f"{val:.0f}%" for val in program_stats['Post_Score_Pct']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Program Type Performance Comparison (All Regions, Ascending by Pre Score)',
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            barmode='group',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 110], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040') # Added category type safety
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Program Analysis by Region (Multi-Select Version)
        st.markdown("---")
        st.subheader("Program Analysis by Region (Multi-Select)")
        
        # Group by Program Type AND Region
        program_region_stats = filtered_df.groupby(['Program Type', 'Region']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        # Calculate percentages
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        # Get list of unique regions in the filtered data
        unique_regions_in_data = sorted(filtered_df['Region'].unique())
        
        # Change selectbox to multiselect
        selected_regions_for_prog = st.multiselect("Select Region(s) to compare Programs", 
                                                   unique_regions_in_data, 
                                                   default=unique_regions_in_data, 
                                                   key='prog_region_viz_select_multi')
        
        fig_prog_region = go.Figure()
        
        # Define a list of distinct colors for the lines
        color_scale = px.colors.qualitative.Plotly 
        color_index = 0

        # Iterate over selected regions and add traces for each
        for region in selected_regions_for_prog:
            region_data = program_region_stats[program_region_stats['Region'] == region]
            # Sort by Program Type for consistent X-axis order
            region_data = region_data.sort_values('Program Type', ascending=True) 

            # Get the base color
            base_color = color_scale[color_index % len(color_scale)]
            
            # Pre-Session Line (dotted)
            fig_prog_region.add_trace(go.Scatter(
                x=region_data['Program Type'],
                y=region_data['Pre_Score_Pct'],
                mode='lines+markers',
                name=f'{region} (Pre)',
                line=dict(color=base_color, width=3, dash='dot'),
                marker=dict(size=10, symbol='circle'),
                hovertemplate = '<b>Program:</b> %{x}<br><b>Pre Score:</b> %{y:.1f}%<br>'
            ))
            
            # Post-Session Line (solid)
            fig_prog_region.add_trace(go.Scatter(
                x=region_data['Program Type'],
                y=region_data['Post_Score_Pct'],
                mode='lines+markers',
                name=f'{region} (Post)',
                line=dict(color=base_color, width=3, dash='solid'),
                marker=dict(size=10, symbol='square'),
                hovertemplate = '<b>Program:</b> %{x}<br><b>Post Score:</b> %{y:.1f}%<br>'
            ))
            
            color_index += 1 # Move to the next color

        # Update Layout
        if selected_regions_for_prog:
            title_regions = ", ".join(selected_regions_for_prog) if len(selected_regions_for_prog) <= 3 else f"{len(selected_regions_for_prog)} Regions"
        else:
            title_regions = "No Regions Selected"
            
        fig_prog_region.update_layout(
            title=f'Program Performance Comparison for {title_regions}',
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_prog_region, use_container_width=True)
        
        # 3. Detailed Program Metrics Table (Existing)
        st.markdown("---")
        st.subheader("📋 Detailed Program Type Metrics")
        display_prog = program_stats.copy()
        display_prog.columns = ['Program Type', 'Avg Pre Score', 'Avg Post Score', 'Total Assessments', 'Pre %', 'Post %', 'Improvement %']
        
        # Select and reorder columns for user display
        display_prog = display_prog[['Program Type', 'Total Assessments', 'Pre %', 'Post %', 'Improvement %']]
        
        # Format columns for display
        display_prog['Pre %'] = display_prog['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_prog['Post %'] = display_prog['Post %'].apply(lambda x: f"{x:.1f}%")
        display_prog['Improvement %'] = display_prog['Improvement %'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_prog, hide_index=True, use_container_width=True)

    # ===== TAB 5: STUDENT PARTICIPATION (UNCHANGED) =====
    with tab5:
        st.header("Student Participation Analysis")
        st.markdown("### Number of Unique Students Taking Assessments")
        
        # Students per Grade
        st.subheader("📚 Students per Grade/Parent Class")
        students_per_grade = filtered_df.groupby('Parent_Class')['Student Id'].nunique().reset_index()
        students_per_grade.columns = ['Grade', 'Number of Students']
        students_per_grade = students_per_grade.sort_values('Grade')
        
        fig_grade = go.Figure()
        fig_grade.add_trace(go.Bar(
            x=students_per_grade['Grade'],
            y=students_per_grade['Number of Students'],
            marker_color='#1abc9c',
            text=students_per_grade['Number of Students'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_grade.update_layout(
            title='Number of Students by Grade',
            xaxis_title='Grade',
            yaxis_title='Number of Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig_grade, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(students_per_grade, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Students Across All Grades", students_per_grade['Number of Students'].sum())
            st.metric("Average Students per Grade", f"{students_per_grade['Number of Students'].mean():.0f}")

        # Students per Region
        st.markdown("---")
        st.subheader("📍 Students per Region")
        students_per_region = filtered_df.groupby('Region')['Student Id'].nunique().reset_index()
        students_per_region.columns = ['Region', 'Number of Students']
        students_per_region = students_per_region.sort_values('Number of Students', ascending=False)
        
        fig_region = go.Figure()
        fig_region.add_trace(go.Bar(
            x=students_per_region['Region'],
            y=students_per_region['Number of Students'],
            marker_color='#e67e22',
            text=students_per_region['Number of Students'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_region.update_layout(
            title='Number of Students by Region',
            xaxis_title='Region',
            yaxis_title='Number of Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040', tickangle=-45)
        )
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Students per Program Type
        st.markdown("---")
        st.subheader("📊 Students per Program Type")
        students_per_program = filtered_df.groupby('Program Type')['Student Id'].nunique().reset_index()
        students_per_program.columns = ['Program Type', 'Number of Students']
        students_per_program = students_per_program.sort_values('Number of Students', ascending=False)
        
        fig_program = go.Figure()
        fig_program.add_trace(go.Bar(
            x=students_per_program['Program Type'],
            y=students_per_program['Number of Students'],
            marker_color='#9b59b6',
            text=students_per_program['Number of Students'],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_program.update_layout(
            title='Number of Students by Program Type',
            xaxis_title='Program Type',
            yaxis_title='Number of Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig_program, use_container_width=True)

        # Cross-Tabulation: Grade vs Region
        st.markdown("---")
        st.subheader("Cross-Tabulation: Students by Grade and Region")
        students_grade_region = filtered_df.groupby(['Parent_Class', 'Region'])['Student Id'].nunique().reset_index()
        students_grade_region.columns = ['Grade', 'Region', 'Number of Students']
        
        # Pivot table
        pivot_grade_region = students_grade_region.pivot(index='Grade', columns='Region', values='Number of Students').fillna(0).astype(int)
        st.dataframe(pivot_grade_region, use_container_width=True)

        # Download all participation data
        st.markdown("---")
        st.subheader("📥 Download Participation Reports")
        col1, col2, col3 = st.columns(3)
        with col1:
            grade_csv = students_per_grade.to_csv(index=False)
            st.download_button("Download Grade Data", grade_csv, "students_per_grade.csv", "text/csv")
        with col2:
            region_csv = students_per_region.to_csv(index=False)
            st.download_button("Download Region Data", region_csv, "students_per_region.csv", "text/csv")
        with col3:
            program_csv = students_per_program.to_csv(index=False)
            st.download_button("Download Program Data", program_csv, "students_per_program.csv", "text/csv")

    # ===== TAB 6: SCHOOL ANALYSIS (UNCHANGED) =====
    with tab6:
        st.header("School Analysis")
        st.markdown("### School Performance and Engagement Metrics")
        try:
            # Group by School Name and UDISE to get unique schools
            # Calculate metrics per school
            school_stats = filtered_df.groupby(['School Name', 'UDISE']).agg({
                'Student Id': 'nunique', # Number of students
                'Content Id': 'nunique', # Number of unique assessments
                'Instructor Login Id': 'nunique', # Number of instructors
                'Region': 'first' # Region (assuming 1 region per school)
            }).reset_index()
            school_stats.columns = ['School Name', 'UDISE', 'Total Students', 'Unique Assessments', 'Total Instructors', 'Region']
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Schools", len(school_stats))
            with col2:
                avg_students = school_stats['Total Students'].mean()
                st.metric("Avg Students/School", f"{avg_students:.1f}")
            with col3:
                avg_assessments = school_stats['Unique Assessments'].mean()
                st.metric("Avg Assessments/School", f"{avg_assessments:.1f}")
            with col4:
                avg_instructors = school_stats['Total Instructors'].mean()
                st.metric("Avg Instructors/School", f"{avg_instructors:.1f}")

            st.markdown("---")
            
            # School List Table
            st.subheader("🏫 Detailed School List")
            
            # Search filter for schools
            search_school = st.text_input("🔍 Search School (Name or UDISE)", "")
            
            # Filter the school stats based on search query
            if search_school:
                search_term = search_school.upper().strip()
                display_school_stats = school_stats[
                    school_stats['School Name'].astype(str).str.upper().str.contains(search_term) |
                    school_stats['UDISE'].astype(str).str.contains(search_term)
                ]
            else:
                display_school_stats = school_stats

            st.dataframe(display_school_stats, hide_index=True, use_container_width=True)
            
            # Download Button
            school_csv = school_stats.to_csv(index=False)
            st.download_button(
                "📥 Download School Data (CSV)",
                school_csv,
                "school_data.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generating school analysis: {str(e)}")

    # ===== TAB 7: DONOR ANALYSIS (UNCHANGED EXCEPT FOR DATA CLEANING) =====
    with tab7:
        st.header("Donor Performance Analysis")
        
        # 1. ADD DONOR FILTER
        # Since df['Donor'] is now uppercase, the list will only contain unique, standardized names.
        all_donors = ['All Donors'] + sorted(filtered_df['Donor'].unique().tolist())
        selected_donor = st.selectbox("Select Donor for Individual Analysis", all_donors)
        
        # Determine the DataFrame to use for the detailed analysis section.
        if selected_donor != 'All Donors':
            # *** SIMPLIFIED FIX: Uses exact match against the master 'df' as the data is now standardized to uppercase in clean_and_process_data ***
            donor_filtered_df = df[ 
                (df['Donor'] == selected_donor) & 
                (df['Region'].isin(selected_regions)) &
                (df['Program Type'].isin(selected_programs)) &
                (df['Parent_Class'].isin(selected_classes))
            ].copy()
            st.subheader(f"Metrics for Donor: **{selected_donor}**")
        else:
            # If 'All Donors' is selected, respect the global sidebar filters (filtered_df)
            donor_filtered_df = filtered_df.copy()
            st.subheader("Metrics for All Donors (Summary View)")

        if donor_filtered_df.empty:
            st.info(f"No data available for the selected donor/filters.")
            # Use continue/return or simply let the rest of the tab not execute data logic
            pass
        else:
            # --- LOGIC FOR ALL DONORS (Summary Table) ---
            # NOTE: This table *must* respect the sidebar filters, so it uses filtered_df
            # 1. Calculate Donor Statistics (Grouping by Donor)
            donor_stats = filtered_df.groupby('Donor').agg(
                Num_Schools=('UDISE', 'nunique'),
                Num_Students=('Student Id', 'nunique'),
                Num_Assessments=('Student Id', 'count'), # Total rows = total assessments
                Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
                Avg_Post_Score_Raw=('Post_Score', 'mean')
            ).reset_index()
            
            # 2. Calculate percentages (These are FLOAT columns)
            donor_stats['Avg Pre Score %'] = (donor_stats['Avg_Pre_Score_Raw'] / 5) * 100
            donor_stats['Avg Post Score %'] = (donor_stats['Avg_Post_Score_Raw'] / 5) * 100
            donor_stats['Improvement %'] = donor_stats['Avg Post Score %'] - donor_stats['Avg Pre Score %']
            
            # 3. Format columns for display - CREATE A COPY FOR DISPLAY
            display_donor_stats = donor_stats.copy()
            # Note: Keeping the donor table sorted descending by Assessments (or by Donor name if needed)
            display_donor_stats = display_donor_stats.sort_values('Num_Assessments', ascending=False)
            
            # Apply formatting
            display_donor_stats['Avg Pre Score %'] = display_donor_stats['Avg Pre Score %'].apply(lambda x: f"{x:.1f}%")
            display_donor_stats['Avg Post Score %'] = display_donor_stats['Avg Post Score %'].apply(lambda x: f"{x:.1f}%")
            display_donor_stats['Improvement %'] = display_donor_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
            
            # Rename columns for display
            display_donor_stats.columns = [
                'Donor', 
                'Total Schools', 
                'Total Students', 
                'Total Assessments', 
                'Avg Pre Score (Raw)', 
                'Avg Post Score (Raw)', 
                'Avg Pre %', 
                'Avg Post %', 
                'Improvement %'
            ]
            
            st.subheader("📋 Donor-wise Summary (Responds to Global Filters)")
            # Show Raw Scores and Percentages
            st.dataframe(display_donor_stats, hide_index=True, use_container_width=True)
            st.markdown("---")

            # --- LOGIC FOR INDIVIDUAL DONOR (Specific Metrics) ---
            # Calculate Metrics for the Selected Donor/All (uses the now correctly filtered donor_filtered_df)
            donor_specific_stats = {
                'Avg Pre Score %': (donor_filtered_df['Pre_Score'].mean() / 5) * 100,
                'Avg Post Score %': (donor_filtered_df['Post_Score'].mean() / 5) * 100,
                'Total Schools': donor_filtered_df['UDISE'].nunique(),
                'Total Students': donor_filtered_df['Student Id'].nunique(),
                'Total Assessments': len(donor_filtered_df)
            }
            donor_specific_stats['Improvement %'] = donor_specific_stats['Avg Post Score %'] - donor_specific_stats['Avg Pre Score %']

            # Key Metrics (Top level summary)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Avg Pre Score", f"{donor_specific_stats['Avg Pre Score %']:.1f}%")
            with col2:
                st.metric("Avg Post Score", f"{donor_specific_stats['Avg Post Score %']:.1f}%")
            with col3:
                st.metric("Improvement", f"{donor_specific_stats['Improvement %']:.1f}%", delta=f"{donor_specific_stats['Improvement %']:.1f}%")
            with col4:
                st.metric("Total Schools", donor_specific_stats['Total Schools'])
            with col5:
                st.metric("Total Students", donor_specific_stats['Total Students'])
            
            st.markdown("---")

            # Breakdown by Region for the Selected Donor
            # NOTE: This calculation uses the donor_filtered_df which is now guaranteed
            # to respect all current filters (global + selected donor)
            st.subheader(f"Region Breakdown for {selected_donor}")
            donor_region_stats = donor_filtered_df.groupby('Region').agg(
                Num_Schools=('UDISE', 'nunique'),
                Num_Students=('Student Id', 'nunique'),
                Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
                Avg_Post_Score_Raw=('Post_Score', 'mean')
            ).reset_index()
            donor_region_stats['Avg Pre %'] = (donor_region_stats['Avg_Pre_Score_Raw'] / 5) * 100
            donor_region_stats['Avg Post %'] = (donor_region_stats['Avg_Post_Score_Raw'] / 5) * 100
            donor_region_stats['Improvement %'] = donor_region_stats['Avg Post %'] - donor_region_stats['Avg Pre %']
            
            # Format and display
            donor_region_stats['Avg Pre %'] = donor_region_stats['Avg Pre %'].apply(lambda x: f"{x:.1f}%")
            donor_region_stats['Avg Post %'] = donor_region_stats['Avg Post %'].apply(lambda x: f"{x:.1f}%")
            donor_region_stats['Improvement %'] = donor_region_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")

            # Select and rename columns for display
            display_donor_region = donor_region_stats[['Region', 'Num_Schools', 'Num_Students', 'Avg Pre %', 'Avg Post %', 'Improvement %']].copy()
            display_donor_region.columns = ['Region', 'Schools', 'Students', 'Avg Pre %', 'Avg Post %', 'Improvement %']
            st.dataframe(display_donor_region, hide_index=True, use_container_width=True)

            # Download Button for Donor-specific analysis (uses float versions)
            st.markdown("---")
            donor_analysis_csv = donor_region_stats.to_csv(index=False)
            st.download_button(
                f"📥 Download {selected_donor} Region Breakdown (CSV)",
                donor_analysis_csv,
                f"{selected_donor}_region_analysis.csv",
                "text/csv"
            )
            
    # ===== TAB 8: SUBJECT ANALYSIS (UNCHANGED) =====
    with tab8:
        if not filtered_df.empty:
            # The function now handles multi-select by default
            subject_stats = tab8_subject_analysis(filtered_df)
        else:
            st.info("No data to display after applying filters.")

    # ===== DOWNLOAD SECTION (UNCHANGED) =====
    st.markdown("---")
    st.subheader("📥 Download Analysis Reports")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Ensure all required dataframes exist before attempting to download
    
    # Region Analysis
    with col1:
        if 'region_stats' in locals():
            region_csv = region_stats.to_csv(index=False)
            st.download_button("Download Region Analysis", region_csv, "region_analysis.csv", "text/csv")
        else:
            st.caption("Region data not available.")

    # Instructor Analysis
    with col2:
        if 'instructor_stats' in locals():
            # Re-sort instructor stats for a consistent download table (descending by Post Score is conventional for ranking)
            instructor_stats_for_download = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
            instructor_csv = instructor_stats_for_download.to_csv(index=False)
            st.download_button("Download Instructor Analysis", instructor_csv, "instructor_analysis.csv", "text/csv")
        else:
            st.caption("Instructor data not available.")

    # Grade Analysis
    with col3:
        if 'grade_stats' in locals():
            # Re-sort grade stats for a consistent download table (by Grade number is conventional)
            grade_stats_for_download = grade_stats.sort_values('Parent_Class')
            grade_csv = grade_stats_for_download.to_csv(index=False)
            st.download_button("Download Grade Analysis", grade_csv, "grade_analysis.csv", "text/csv")
        else:
            st.caption("Grade data not available.")

    # Program Type Analysis
    with col4:
        if 'program_stats' in locals():
            # Re-sort program stats for a consistent download table (by Program Type is conventional)
            program_stats_for_download = program_stats.sort_values('Program Type')
            program_csv = program_stats_for_download.to_csv(index=False)
            st.download_button("Download Program Analysis", program_csv, "program_analysis.csv", "text/csv")
        else:
            st.caption("Program data not available.")
            
    # Donor Summary
    with col5:
        if 'donor_stats' in locals():
            donor_summary_csv = donor_stats.to_csv(index=False)
            st.download_button("Download Donor Summary", donor_summary_csv, "donor_summary.csv", "text/csv")
        else:
            st.caption("Donor data not available.")

    # Subject Analysis (Handled by the function return)
    with col6:
        if subject_stats is not None:
            # Already handled within the tab8 function, just need to confirm it ran
            st.caption("See Tab 8 for Subject Download.")
        else:
            st.caption("Subject data not available.")


else:
    st.info("👆 Please upload your student data Excel file to begin")
    
    st.markdown("---")
    st.subheader("📋 Required Excel Columns")
    st.markdown("""
    Your Excel file must contain these columns:
    
    **Identification Columns:**
    - `Region` - Geographic region
    - `School Name` - Name of the school
    - `Donor` - The donor/partner associated with the record
    - **`Subject` - The subject name**
    - `UDISE` - School unique ID
    - `Student Id` - Unique student identifier
    - `Class` - Class with section (e.g., 6-A, 7-B)
    - `Instructor Name` - Name of instructor
    - `Instructor Login Id` - Login ID of instructor
    - `Program Type` - Program type code
    - `Content Id` - Assessment/Content Identifier
    - `Date_Post` - Assessment Date (Required for tracking unique tests)
    
    **Pre-Session (Questions & Answers):**
    - `Q1`, `Q2`, `Q3`, `Q4`, `Q5` - Student responses
    - `Q1 Answer`, `Q2 Answer`, `Q3 Answer`, `Q4 Answer`, `Q5 Answer` - Correct answers
    
    **Post-Session (Questions & Answers):**
    - `Q1_Post`, `Q2_Post`, `Q3_Post`, `Q4_Post`, `Q5_Post` - Student responses
    - `Q1_Answer_Post`, `Q2_Answer_Post`, `Q3_Answer_Post`, `Q4_Answer_Post`, `Q5_Answer_Post` - Correct answers
    
    ---
    
    **What happens when you upload:**
    1. ✅ Data is cleaned (incomplete records removed)
    2. ✅ Scores are calculated automatically
    3. ✅ Program types are standardized
    4. ✅ **Donor names are standardized to UPPERCASE (Fix for 'Adobe' vs 'ADOBE')**
    5. ✅ **Subject names are standardized (e.g., 'Science' and 'SCIENCE' become 'SCIENCE')**
    6. ✅ Interactive dashboard is generated
    7. ✅ Download options for all analyses
    """)