import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Student Assessment Dashboard", layout="wide", page_icon="📊")

# ===== DATA CLEANING FUNCTIONS (UNCHANGED) =====
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
    
    return df, initial_count, cleaned_count

# ===== TAB 8: SUBJECT ANALYSIS (MODIFIED) =====
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
    
    unique_regions = sorted(df['Region'].unique())
    selected_region_for_subject = st.selectbox("Select Region for Subject Breakdown", unique_regions, key='region_subject_select')

    # Filter data for the selected region
    region_subject_data = subject_region_stats[subject_region_stats['Region'] == selected_region_for_subject].copy()
    
    # Sort in ascending order of PRE Score %
    region_subject_data = region_subject_data.sort_values('Pre_Score_Pct', ascending=True)

    fig_subj_region = go.Figure()
    
    fig_subj_region.add_trace(go.Scatter(
        x=region_subject_data['Subject'],
        y=region_subject_data['Pre_Score_Pct'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#8e44ad', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in region_subject_data['Pre_Score_Pct']],
        textposition='top center'
    ))
    
    fig_subj_region.add_trace(go.Scatter(
        x=region_subject_data['Subject'],
        y=region_subject_data['Post_Score_Pct'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#f39c12', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in region_subject_data['Post_Score_Pct']],
        textposition='top center'
    ))
    
    fig_subj_region.update_layout(
        title=f'Subject Performance in **{selected_region_for_subject}** (Ascending by Pre Score)',
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
    
    unique_subjects = sorted(df['Subject'].unique())
    selected_subject_for_region = st.selectbox("Select Subject for Region Breakdown", unique_subjects, key='subject_region_select')

    # Filter data for the selected subject
    subject_region_data = subject_region_stats[subject_region_stats['Subject'] == selected_subject_for_region].copy()
    
    # Sort in ascending order of PRE Score %
    subject_region_data = subject_region_data.sort_values('Pre_Score_Pct', ascending=True)

    fig_region_subj = go.Figure()
    
    fig_region_subj.add_trace(go.Scatter(
        x=subject_region_data['Region'],
        y=subject_region_data['Pre_Score_Pct'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#1abc9c', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_region_data['Pre_Score_Pct']],
        textposition='top center'
    ))
    
    fig_region_subj.add_trace(go.Scatter(
        x=subject_region_data['Region'],
        y=subject_region_data['Post_Score_Pct'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in subject_region_data['Post_Score_Pct']],
        textposition='top center'
    ))
    
    fig_region_subj.update_layout(
        title=f'Region Performance in **{selected_subject_for_region}** (Ascending by Pre Score)',
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

# ===== MAIN APPLICATION (MODIFIED TO FIX ERRORS) =====

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
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Region filter
    all_regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", all_regions)
    
    # Program Type filter
    all_programs = ['All'] + sorted(df['Program Type'].unique().tolist())
    selected_program = st.sidebar.selectbox("Select Program Type", all_programs)
    
    # Parent Class filter
    all_classes = ['All'] + sorted(df['Parent_Class'].unique().tolist())
    selected_class = st.sidebar.selectbox("Select Grade", all_classes)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_program != 'All':
        filtered_df = filtered_df[filtered_df['Program Type'] == selected_program]
    if selected_class != 'All':
        filtered_df = filtered_df[filtered_df['Parent_Class'] == selected_class]
    
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

    # ===== TAB 1: REGION ANALYSIS (UNCHANGED) =====
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
        
        # Region by Program Type
        st.subheader("Region Analysis by Program Type")
        
        program_region_stats = filtered_df.groupby(['Region', 'Program Type']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        selected_program_type = st.selectbox("Select Program Type for Detailed View", 
                                             sorted(filtered_df['Program Type'].unique()))
        
        prog_data = program_region_stats[program_region_stats['Program Type'] == selected_program_type]
        # Sort in ascending order of PRE Score Pct
        prog_data = prog_data.sort_values('Pre_Score_Pct', ascending=True)

        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=prog_data['Region'],
            y=prog_data['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in prog_data['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.add_trace(go.Scatter(
            x=prog_data['Region'],
            y=prog_data['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in prog_data['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig2.update_layout(
            title=f'{selected_program_type} - Region-wise Performance (Ascending by Pre Score)',
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040') # Force category type
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
    
    # ===== TAB 2: INSTRUCTOR ANALYSIS (FIXED TOTAL INSTRUCTORS COUNT) =====
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
            Region=('Region', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])  # Most common region
        ).reset_index()
        
        all_instructors.columns = ['Instructor Name', 'Instructor Login Id', 'Number of Assessments', 'Total Students', 'Primary Region']
        all_instructors = all_instructors.sort_values('Number of Assessments', ascending=False)
        
        # Search functionality
        search_instructor = st.text_input("🔍 Search for an instructor (Name or ID)", "")
        
        if search_instructor:
            filtered_instructors = all_instructors[
                all_instructors['Instructor Name'].str.contains(search_instructor, case=False, na=False) |
                all_instructors['Instructor Login Id'].astype(str).str.contains(search_instructor, case=False, na=False)
            ]
        else:
            filtered_instructors = all_instructors
        
        # FIX APPLIED: Calculate unique instructor count directly and use it for all metrics.
        total_unique_instructors = filtered_df['Instructor Name'].nunique()
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            # FIX: Use the explicit unique count
            st.metric("Total Instructors", total_unique_instructors) 
        with col2:
            st.metric("Avg Assessments per Instructor", f"{all_instructors['Number of Assessments'].mean():.1f}")
        with col3:
            st.metric("Max Assessments by One Instructor", all_instructors['Number of Assessments'].max())
        
        # Display the full table
        st.dataframe(
            filtered_instructors,
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Download option for instructor assessment data
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
    
    # ===== TAB 3: GRADE ANALYSIS (FIXED STRING FORMATTING & SORTING) =====
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
            textfont=dict(size=14)
        ))
        
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14)
        ))
        
        # CRITICAL FIX: Explicitly set type='category' for x-axis.
        # Otherwise, Plotly interprets '6', '7', '8' as numbers and auto-sorts them numerically, 
        # destroying the score-based sorting order.
        fig.update_layout(
            title='Grade-wise Performance Comparison (Ascending by Pre Score)',
            xaxis_title='Grade',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040') # Force category order
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Grade statistics table
        st.subheader("Detailed Grade Statistics")
        # Create a copy for DISPLAY formatting only
        display_stats = grade_stats.copy() 
        
        # Rename the columns for display clarity
        display_stats.columns = ['Grade', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        
        # Select and reorder columns
        display_stats = display_stats[['Grade', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        
        # APPLY STRING FORMATTING (This line should be the only place where the float columns become strings)
        display_stats['Pre %'] = display_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Post %'] = display_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_stats['Improvement %'] = display_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_stats, hide_index=True, use_container_width=True)
    
    # ===== TAB 4: PROGRAM TYPE ANALYSIS (MODIFIED SORTING) =====
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
        
        # 2. Program Analysis by Region
        st.markdown("---")
        st.subheader("Program Analysis by Region")
        
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

        # Selectbox to choose Region
        selected_region_for_prog = st.selectbox("Select Region to compare Programs", 
                                             unique_regions_in_data, key='prog_region_viz_select')
        
        # Filter data based on selection
        region_data = program_region_stats[program_region_stats['Region'] == selected_region_for_prog].copy()
        
        # Sort in ascending order of PRE Score Pct
        region_data = region_data.sort_values('Pre_Score_Pct', ascending=True)

        # Create Line/Marker Chart (Similar to Tab 1 style)
        fig_prog_region = go.Figure()
        
        fig_prog_region.add_trace(go.Scatter(
            x=region_data['Program Type'],
            y=region_data['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#1abc9c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_data['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig_prog_region.add_trace(go.Scatter(
            x=region_data['Program Type'],
            y=region_data['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#e67e22', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_data['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig_prog_region.update_layout(
            title=f'Program Performance in {selected_region_for_prog} (Ascending by Pre Score)',
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            height=450,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 110], gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040') # Force category type
        )
        
        st.plotly_chart(fig_prog_region, use_container_width=True)
        
        # 3. Program stats table (Existing)
        st.subheader("Detailed Program Type Statistics")
        # Create a copy for DISPLAY formatting only
        display_prog = program_stats.copy()
        
        # Rename the columns for display clarity
        display_prog.columns = ['Program', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        
        # Select and reorder columns
        display_prog = display_prog[['Program', 'Pre %', 'Post %', 'Improvement %', 'Students']]
        
        # APPLY STRING FORMATTING
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(students_per_region, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Regions", len(students_per_region))
            st.metric("Average Students per Region", f"{students_per_region['Number of Students'].mean():.0f}")
        
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(students_per_program, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Program Types", len(students_per_program))
            st.metric("Average Students per Program", f"{students_per_program['Number of Students'].mean():.0f}")
        
        # Combined breakdown: Region x Program Type
        st.markdown("---")
        st.subheader("🔄 Students by Region and Program Type")
        
        students_region_program = filtered_df.groupby(['Region', 'Program Type'])['Student Id'].nunique().reset_index()
        students_region_program.columns = ['Region', 'Program Type', 'Number of Students']
        
        # Pivot table for better visualization
        pivot_table = students_region_program.pivot(index='Region', columns='Program Type', values='Number of Students').fillna(0).astype(int)
        
        st.dataframe(pivot_table, use_container_width=True)
        
        # Combined breakdown: Grade x Region
        st.markdown("---")
        st.subheader("🔄 Students by Grade and Region")
        
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
                'Student Id': 'nunique',        # Number of students
                'Content Id': 'nunique',        # Number of unique assessments
                'Instructor Login Id': 'nunique', # Number of instructors
                'Region': 'first'               # Region (assuming 1 region per school)
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
            if search_school:
                display_schools = school_stats[
                    school_stats['School Name'].str.contains(search_school, case=False, na=False) |
                    school_stats['UDISE'].astype(str).str.contains(search_school, case=False, na=False)
                ]
            else:
                display_schools = school_stats
                
            st.dataframe(display_schools, hide_index=True, use_container_width=True)
            
            # Download button for school data
            school_csv = school_stats.to_csv(index=False)
            st.download_button(
                "📥 Download School Analysis Data", 
                school_csv, 
                "school_analysis.csv", 
                "text/csv"
            )
            
            st.markdown("---")
            
            # Graph: Number of Schools per Region
            st.subheader("📊 Number of Schools per Region")
            
            schools_per_region = school_stats.groupby('Region')['School Name'].count().reset_index()
            schools_per_region.columns = ['Region', 'Number of Schools']
            schools_per_region = schools_per_region.sort_values('Number of Schools', ascending=False)
            
            fig_schools = go.Figure()
            fig_schools.add_trace(go.Bar(
                x=schools_per_region['Region'],
                y=schools_per_region['Number of Schools'],
                marker_color='#8e44ad',
                text=schools_per_region['Number of Schools'],
                textposition='outside',
                textfont=dict(size=14, color='white')
            ))
            
            fig_schools.update_layout(
                title='Count of Unique Schools by Region',
                xaxis_title='Region',
                yaxis_title='Number of Schools',
                height=450,
                plot_bgcolor='#2b2b2b',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                yaxis=dict(gridcolor='#404040'),
                xaxis=dict(gridcolor='#404040')
            )
            
            st.plotly_chart(fig_schools, use_container_width=True)
            
        except KeyError as e:
            st.error(f"Missing required columns for School Analysis: {e}")
            st.warning("Please ensure your Excel file contains 'School Name' and 'UDISE' columns.")
            school_stats = None # Set to None if error occurs to handle download later

    # ===== TAB 7: DONOR ANALYSIS (FIXED STRING FORMATTING) =====
    with tab7:
        st.header("Donor Performance Analysis")
        
        # 1. ADD DONOR FILTER
        all_donors = ['All Donors'] + sorted(filtered_df['Donor'].unique().tolist())
        selected_donor = st.selectbox("Select Donor for Individual Analysis", all_donors)
        
        # Apply the donor filter to create donor_filtered_df
        if selected_donor != 'All Donors':
            donor_filtered_df = filtered_df[filtered_df['Donor'] == selected_donor]
            st.subheader(f"Metrics for Donor: **{selected_donor}**")
        else:
            donor_filtered_df = filtered_df
            st.subheader("Metrics for All Donors (Summary View)")

        if donor_filtered_df.empty:
            st.info(f"No data available for the selected donor/filters.")
            # Use continue/return or simply let the rest of the tab not execute data logic
            pass 
        
        # --- LOGIC FOR ALL DONORS (Summary Table) ---
        
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
        
        # Select and rename final columns
        display_donor_stats = display_donor_stats[[
            'Donor', 
            'Num_Schools', 
            'Num_Students', 
            'Num_Assessments', 
            'Avg Pre Score %', 
            'Avg Post Score %', 
            'Improvement %'
        ]]
        
        display_donor_stats.columns = [
            'Donor', 
            'Schools', 
            'Students', 
            'Assessments', 
            'Avg Pre %', 
            'Avg Post %', 
            'Improvement %'
        ]
        
        # Apply string formatting (AFTER all calculations/renaming on the DISPLAY COPY)
        display_donor_stats['Avg Pre %'] = display_donor_stats['Avg Pre %'].apply(lambda x: f"{x:.1f}%")
        display_donor_stats['Avg Post %'] = display_donor_stats['Avg Post %'].apply(lambda x: f"{x:.1f}%")
        display_donor_stats['Improvement %'] = display_donor_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.subheader("Detailed Donor Analysis Table (All Donors)")
        st.dataframe(display_donor_stats, hide_index=True, use_container_width=True)
        st.markdown("---")
        
        # --- LOGIC FOR INDIVIDUAL DONOR (Specific Metrics) ---

        if not donor_filtered_df.empty:
            
            # Calculate Metrics for the Selected Donor/All
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
            
            # Create a copy for DISPLAY
            display_donor_region = donor_region_stats.copy()
            display_donor_region = display_donor_region[[
                'Region', 
                'Num_Schools', 
                'Num_Students', 
                'Avg Pre %', 
                'Avg Post %', 
                'Improvement %'
            ]]
            
            display_donor_region.columns = ['Region', 'Schools', 'Students', 'Avg Pre %', 'Avg Post %', 'Improvement %']
            
            # Apply string formatting
            display_donor_region['Avg Pre %'] = display_donor_region['Avg Pre %'].apply(lambda x: f"{x:.1f}%")
            display_donor_region['Avg Post %'] = display_donor_region['Avg Post %'].apply(lambda x: f"{x:.1f}%")
            display_donor_region['Improvement %'] = display_donor_region['Improvement %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_donor_region, hide_index=True, use_container_width=True)

            # Download button for ALL DONOR SUMMARY TABLE (unchanged, using pre-calculated donor_stats)
            st.markdown("---")
            # Must use the float columns for download, not the display string columns
            donor_csv_for_download = donor_stats.copy()
            donor_csv_for_download = donor_csv_for_download.sort_values('Num_Assessments', ascending=False)
            donor_csv_for_download.columns = [
                'Donor', 
                'Schools', 
                'Students', 
                'Assessments', 
                'Avg_Pre_Score_Raw', 
                'Avg_Post_Score_Raw', 
                'Avg Pre %', 
                'Avg Post %', 
                'Improvement %'
            ]
            donor_csv = donor_csv_for_download.to_csv(index=False)
            st.download_button(
                "📥 Download All Donor Analysis Data (CSV)",
                donor_csv,
                "donor_analysis.csv",
                "text/csv"
            )

        else:
            # If donor_filtered_df is empty, show the total summary and inform the user
            st.info("No records match the current filter selection.")
            
    # ===== TAB 8: SUBJECT ANALYSIS (NEW/MODIFIED) =====
    with tab8:
        # Call the new function
        if not filtered_df.empty:
            subject_stats = tab8_subject_analysis(filtered_df)
        else:
            st.info("No data to display after applying filters.")

    
    # ===== DOWNLOAD SECTION (UNCHANGED) =====
    st.markdown("---")
    st.subheader("📥 Download Analysis Reports")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        region_csv = region_stats.to_csv(index=False)
        st.download_button("Download Region Analysis", region_csv, "region_analysis.csv", "text/csv")
    
    with col2:
        # Re-sort instructor stats for a consistent download table (descending by Post Score is conventional for ranking)
        instructor_stats_for_download = instructor_stats.sort_values('Post_Score_Pct', ascending=False)
        instructor_csv = instructor_stats_for_download.to_csv(index=False)
        st.download_button("Download Instructor Analysis", instructor_csv, "instructor_analysis.csv", "text/csv")
    
    with col3:
        # Re-sort grade stats for a consistent download table (by Grade number is conventional)
        grade_stats_for_download = grade_stats.sort_values('Parent_Class')
        grade_csv = grade_stats_for_download.to_csv(index=False)
        st.download_button("Download Grade Analysis", grade_csv, "grade_analysis.csv", "text/csv")

    with col4:
        # Re-sort program stats for a consistent download table (alphabetical by program type is conventional)
        program_stats_for_download = program_stats.sort_values('Program Type')
        program_csv = program_stats_for_download.to_csv(index=False)
        st.download_button("Download Program Analysis", program_csv, "program_analysis.csv", "text/csv")
            
    with col5:
        # Check if school_stats exists (it's created inside the tab)
        if 'school_stats' in locals() and school_stats is not None:
            school_csv = school_stats.to_csv(index=False)
            st.download_button("Download School Analysis", school_csv, "school_analysis_summary.csv", "text/csv")

    with col6:
        # Check if subject_stats was successfully generated in Tab 8
        if subject_stats is not None:
            # Prepare subject stats for download (using the float columns from the start of tab 8)
            subject_csv_final = subject_stats[[
                'Subject', 
                'Num_Students', 
                'Num_Assessments', 
                'Avg Pre Score %', 
                'Avg Post Score %', 
                'Improvement %'
            ]].copy()
            subject_csv_final.columns = ['Subject', 'Unique Students', 'Total Assessments', 'Avg Pre %', 'Avg Post %', 'Improvement %']
            subject_csv_final = subject_csv_final.sort_values('Subject')
            final_csv_output = subject_csv_final.to_csv(index=False)
            st.download_button("Download Subject Analysis", final_csv_output, "subject_analysis_summary.csv", "text/csv")


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
    4. ✅ Interactive dashboard is generated
    5. ✅ Download options for all analyses
    """)