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
    
    # Sort in ascending order of Pre Score % (MODIFIED)
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
        title='Subject-wise Pre and Post Assessment Scores (Ascending by Pre Score)', # MODIFIED TITLE
        xaxis_title='Subject',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(gridcolor='#404040')
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
    region_subject_data = region_subject_data.sort_values('Pre_Score_Pct', ascending=True) # MODIFIED

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
        title=f'Subject Performance in **{selected_region_for_subject}** (Ascending by Pre Score)', # MODIFIED TITLE
        xaxis_title='Subject',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(gridcolor='#404040')
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
    subject_region_data = subject_region_data.sort_values('Pre_Score_Pct', ascending=True) # MODIFIED

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
        title=f'Region Performance in **{selected_subject_for_region}** (Ascending by Pre Score)', # MODIFIED TITLE
        xaxis_title='Region',
        yaxis_title='Average Score (%)',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(gridcolor='#404040')
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
            
            # Basic checks for required columns (FIX 1: Corrected undefined variable required_df)
            required_check_cols = ['Date_Post', 'Donor', 'Subject', 'Region', 'Student Id', 'Class', 'Program Type', 'Q1', 'Q1_Post']
            missing_cols = [col for col in required_check_cols if col not in raw_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}. Please add these columns and try again.")
                st.stop()
                
            df, initial_count, cleaned_count = clean_and_process_data(raw_df)
            
            # Show cleaning summary
            st.success("✅ Data cleaned successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Records", initial_count)
            with col2:
                st.metric("Records Removed", initial_count - cleaned_count)
            with col3:
                st.metric("Final Records", cleaned_count)
            
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

    # ===== TAB 1: REGION ANALYSIS (MODIFIED) =====
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
        
        # Sort in ascending order of Pre Score Pct (MODIFIED)
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
            title='Region-wise Performance Comparison (Ascending by Pre Score)', # MODIFIED TITLE
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            hovermode='x unified',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
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
        # Sort in ascending order of Pre Score Pct (MODIFIED)
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
            title=f'{selected_program_type} - Region-wise Performance (Ascending by Pre Score)', # MODIFIED TITLE
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
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
    
    # ===== TAB 2: INSTRUCTOR ANALYSIS (MODIFIED) =====
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
        
        # Get top N performers, but sort them ascending by Pre Score for the plot (MODIFIED)
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
            title=f'Top {top_n} Instructors by Post-Session Performance (Ascending by Pre Score)', # MODIFIED TITLE
            xaxis_title='Instructor',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(tickangle=-45, gridcolor='#404040')
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
            filtered_df['Content Id'].astype(str) + '_' + filtered_df['Class'].astype(str) + '_' + filtered_df['School Name'].fillna('NA').astype(str) + '_' + filtered_df['Date_Post'].astype(str) # Added Date_Post
        )
        
        # Calculate number of assessments (using the session key) per instructor
        all_instructors = filtered_df.groupby(['Instructor Name', 'Instructor Login Id']).agg({
            'Assessment_Session_Key': 'nunique', # This correctly counts distinct sessions
            'Student Id': 'count',
            'Region': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0] # Most common region
        }).reset_index()
        
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

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Instructors", len(all_instructors))
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
            st.metric("Total Unique Instructors", filtered_df['Instructor Name'].nunique())
            st.metric("Average per Region", f"{instructors_per_region['Number of Instructors'].mean():.1f}")

    # ===== TAB 3: GRADE ANALYSIS (MODIFIED) =====
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
        
        # Sort in ascending order of Pre Score Pct (MODIFIED)
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
        
        fig.update_layout(
            title='Grade-wise Performance Comparison (Ascending by Pre Score)', # MODIFIED TITLE
            xaxis_title='Grade',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040'),
            xaxis=dict(gridcolor='#404040')
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

    # ===== TAB 4: PROGRAM TYPE ANALYSIS (MODIFIED) =====
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
        
        # Sort by Post Score % descending for the table
        program_stats_table = program_stats.sort_values('Post_Score_Pct', ascending=False)

        st.subheader("Performance Comparison (Pre vs. Post)")
        
        # Sort ascending by Pre Score % for the plot
        program_stats_plot = program_stats.sort_values('Pre_Score_Pct', ascending=True)

        fig_prog = go.Figure()
        
        fig_prog.add_trace(go.Scatter(
            x=program_stats_plot['Program Type'],
            y=program_stats_plot['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#8e44ad', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in program_stats_plot['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig_prog.add_trace(go.Scatter(
            x=program_stats_plot['Program Type'],
            y=program_stats_plot['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in program_stats_plot['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig_prog.update_layout(
            title='Program Type Performance Comparison (Ascending by Pre Score)', # MODIFIED TITLE
            xaxis_title='Program Type',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
        )
        
        st.plotly_chart(fig_prog, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Program Type Analysis - Region Breakdown (Existing)
        program_region_stats = filtered_df.groupby(['Program Type', 'Region']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        st.subheader("Region Breakdown for Selected Program Type")
        selected_program_type_for_region = st.selectbox("Select Program Type for Region Breakdown", sorted(filtered_df['Program Type'].unique()), key='prog_type_region')
        
        region_data = program_region_stats[program_region_stats['Program Type'] == selected_program_type_for_region].copy()
        # Sort in ascending order of Pre Score Pct for the line chart (MODIFIED)
        region_data = region_data.sort_values('Pre_Score_Pct', ascending=True)

        fig_region_prog = go.Figure()
        
        fig_region_prog.add_trace(go.Scatter(
            x=region_data['Region'],
            y=region_data['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#1abc9c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_data['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig_region_prog.add_trace(go.Scatter(
            x=region_data['Region'],
            y=region_data['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in region_data['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig_region_prog.update_layout(
            title=f'Region Performance for **{selected_program_type_for_region}** (Ascending by Pre Score)', # MODIFIED TITLE
            xaxis_title='Region',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
        )
        
        st.plotly_chart(fig_region_prog, use_container_width=True)
        
        # Program Type Statistics Table
        st.markdown("---")
        st.subheader("Detailed Program Type Statistics")
        
        # Prepare table data
        display_program_stats = program_stats_table.copy()
        display_program_stats.columns = ['Program Type', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_program_stats = display_program_stats[['Program Type', 'Pre %', 'Post %', 'Improvement %', 'Students']]

        display_program_stats['Pre %'] = display_program_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_program_stats['Post %'] = display_program_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_program_stats['Improvement %'] = display_program_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_program_stats, hide_index=True, use_container_width=True)

        # Download option
        program_csv = program_stats.to_csv(index=False)
        st.download_button(
            "📥 Download Program Type Analysis Data (CSV)",
            program_csv,
            "program_type_analysis.csv",
            "text/csv"
        )

    # ===== TAB 5: STUDENT PARTICIPATION (UNCHANGED) =====
    with tab5:
        st.header("Student Participation Analysis")

        # 1. Total Student Count by Region
        student_counts_region = filtered_df.groupby('Region')['Student Id'].nunique().reset_index()
        student_counts_region.columns = ['Region', 'Unique Students']
        student_counts_region = student_counts_region.sort_values('Unique Students', ascending=False)

        st.subheader("Unique Student Count by Region")
        fig_student_region = go.Figure(go.Bar(
            x=student_counts_region['Region'],
            y=student_counts_region['Unique Students'],
            marker_color=px.colors.sequential.Teal,
            text=student_counts_region['Unique Students'],
            textposition='outside'
        ))
        
        fig_student_region.update_layout(
            title='Unique Students Counted Per Region',
            xaxis_title='Region',
            yaxis_title='Unique Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig_student_region, use_container_width=True)
        
        # Table and Metrics
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(student_counts_region, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Unique Students (Filtered)", filtered_df['Student Id'].nunique())
            st.metric("Average Students per Region", f"{student_counts_region['Unique Students'].mean():.0f}")
        
        st.markdown("---")
        
        # 2. Total Student Count by Program Type
        student_counts_program = filtered_df.groupby('Program Type')['Student Id'].nunique().reset_index()
        student_counts_program.columns = ['Program Type', 'Unique Students']
        student_counts_program = student_counts_program.sort_values('Unique Students', ascending=False)
        
        st.subheader("Unique Student Count by Program Type")
        fig_student_program = go.Figure(go.Bar(
            x=student_counts_program['Program Type'],
            y=student_counts_program['Unique Students'],
            marker_color=px.colors.sequential.Plasma,
            text=student_counts_program['Unique Students'],
            textposition='outside'
        ))
        
        fig_student_program.update_layout(
            title='Unique Students Counted Per Program Type',
            xaxis_title='Program Type',
            yaxis_title='Unique Students',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040')
        )
        st.plotly_chart(fig_student_program, use_container_width=True)
        
        # Table and Metrics
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(student_counts_program, hide_index=True, use_container_width=True)
        with col2:
            st.metric("Total Unique Students (Filtered)", filtered_df['Student Id'].nunique())
            st.metric("Average Students per Program", f"{student_counts_program['Unique Students'].mean():.0f}")

    # ===== TAB 6: SCHOOL ANALYSIS (UNCHANGED) =====
    with tab6:
        st.header("School-wise Performance and Participation Analysis")
        
        # School-wise performance
        school_stats = filtered_df.groupby('School Name').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'nunique',
            'Region': 'first' 
        }).reset_index()
        
        school_stats['Pre_Score_Pct'] = (school_stats['Pre_Score'] / 5) * 100
        school_stats['Post_Score_Pct'] = (school_stats['Post_Score'] / 5) * 100
        school_stats['Improvement'] = school_stats['Post_Score_Pct'] - school_stats['Pre_Score_Pct']
        
        # Rename columns for display
        school_stats.columns = ['School Name', 'Pre Score Raw', 'Post Score Raw', 'Unique Students', 'Region', 'Pre %', 'Post %', 'Improvement %']
        
        # Sort by Post Score % Descending for ranking table
        school_stats = school_stats.sort_values('Post %', ascending=False)

        st.subheader("🏆 Top Performing Schools (By Post Score)")
        top_n_schools = st.slider("Number of top schools to display", 5, 25, 10, key='top_schools_slider')
        
        top_schools_display = school_stats.head(top_n_schools).copy()
        
        # Formatting
        top_schools_display['Pre %'] = top_schools_display['Pre %'].apply(lambda x: f"{x:.1f}%")
        top_schools_display['Post %'] = top_schools_display['Post %'].apply(lambda x: f"{x:.1f}%")
        top_schools_display['Improvement %'] = top_schools_display['Improvement %'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            top_schools_display[['School Name', 'Region', 'Post %', 'Improvement %', 'Unique Students']], 
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("---")
        
        # School Participation (Number of assessments)
        st.subheader("📊 School Participation (Assessment Count)")
        
        # Using the Assessment_Session_Key calculated in TAB 2 (re-calculate if needed, but safer to assume it exists if data exists)
        if 'Assessment_Session_Key' not in filtered_df.columns:
            filtered_df['Assessment_Session_Key'] = (
                filtered_df['Content Id'].astype(str) + '_' + filtered_df['Class'].astype(str) + '_' + filtered_df['School Name'].fillna('NA').astype(str) + '_' + filtered_df['Date_Post'].astype(str)
            )

        school_participation = filtered_df.groupby('School Name').agg({
            'Assessment_Session_Key': 'nunique',
            'Region': 'first'
        }).reset_index()
        
        school_participation.columns = ['School Name', 'Total Assessments', 'Region']
        school_participation = school_participation.sort_values('Total Assessments', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(school_participation, hide_index=True, use_container_width=True, height=300)
        with col2:
            st.metric("Total Unique Schools", len(school_participation))
            st.metric("Max Assessments by a School", school_participation['Total Assessments'].max())
            st.metric("Avg Assessments per School", f"{school_participation['Total Assessments'].mean():.1f}")

    # ===== TAB 7: DONOR ANALYSIS (UNCHANGED) =====
    with tab7:
        st.header("Donor-wise Performance Analysis")
        
        # Donor performance analysis
        donor_stats = filtered_df.groupby('Donor').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count'
        }).reset_index()
        
        donor_stats['Pre_Score_Pct'] = (donor_stats['Pre_Score'] / 5) * 100
        donor_stats['Post_Score_Pct'] = (donor_stats['Post_Score'] / 5) * 100
        donor_stats['Improvement'] = donor_stats['Post_Score_Pct'] - donor_stats['Pre_Score_Pct']
        
        # Sort by Post Score % descending for the table
        donor_stats_table = donor_stats.sort_values('Post_Score_Pct', ascending=False)

        st.subheader("Performance Comparison (Pre vs. Post)")
        
        # Sort ascending by Pre Score % for the plot
        donor_stats_plot = donor_stats.sort_values('Pre_Score_Pct', ascending=True)

        fig_donor = go.Figure()
        
        fig_donor.add_trace(go.Scatter(
            x=donor_stats_plot['Donor'],
            y=donor_stats_plot['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session Average',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in donor_stats_plot['Pre_Score_Pct']],
            textposition='top center'
        ))
        
        fig_donor.add_trace(go.Scatter(
            x=donor_stats_plot['Donor'],
            y=donor_stats_plot['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session Average',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10),
            text=[f"{val:.0f}%" for val in donor_stats_plot['Post_Score_Pct']],
            textposition='top center'
        ))
        
        fig_donor.update_layout(
            title='Donor-wise Performance Comparison (Ascending by Pre Score)', # MODIFIED TITLE (Though it was not explicitly requested, it is a line graph and follows the pattern)
            xaxis_title='Donor',
            yaxis_title='Average Score (%)',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(range=[0, 100], gridcolor='#404040')
        )
        
        st.plotly_chart(fig_donor, use_container_width=True)
        
        st.markdown("---")

        st.subheader("Detailed Donor Statistics")
        
        # Prepare table data
        display_donor_stats = donor_stats_table.copy()
        display_donor_stats.columns = ['Donor', 'Pre Score', 'Post Score', 'Students', 'Pre %', 'Post %', 'Improvement %']
        display_donor_stats = display_donor_stats[['Donor', 'Pre %', 'Post %', 'Improvement %', 'Students']]

        display_donor_stats['Pre %'] = display_donor_stats['Pre %'].apply(lambda x: f"{x:.1f}%")
        display_donor_stats['Post %'] = display_donor_stats['Post %'].apply(lambda x: f"{x:.1f}%")
        display_donor_stats['Improvement %'] = display_donor_stats['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_donor_stats, hide_index=True, use_container_width=True)

        # Download option
        donor_csv = donor_stats.to_csv(index=False)
        st.download_button(
            "📥 Download Donor Analysis Data (CSV)",
            donor_csv,
            "donor_analysis.csv",
            "text/csv"
        )

    # ===== TAB 8: SUBJECT ANALYSIS (MODIFIED) =====
    with tab8:
        # Pass the filtered DataFrame to the function
        subject_stats = tab8_subject_analysis(filtered_df)

else:
    st.info("☝️ Please upload an Excel file to start the analysis.")
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
    """)