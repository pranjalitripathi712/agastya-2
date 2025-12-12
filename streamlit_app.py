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
    pd.DataFrame: Cleaned and processed dataframe, initial and cleaned counts, 
                  initial pre-test count, and initial post-test count.
    """
    
    initial_count = len(df)
    
    # ===== STEP 1: DATA CLEANING =====
    # Define pre and post question columns
    pre_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    post_questions = ['Q1_Post', 'Q2_Post', 'Q3_Post', 'Q4_Post', 'Q5_Post']
    
    # --- NEW ADDITION: Calculate initial test counts before cleaning ---
    # Count rows where ANY pre-question column is NOT NA
    pre_test_count_initial = df[pre_questions].notna().any(axis=1).sum()
    # Count rows where ANY post-question column is NOT NA
    post_test_count_initial = df[post_questions].notna().any(axis=1).sum()
    # -----------------------------------------------------------------

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
        }, regex=False)

    # ===== STEP 7: STANDARDIZE DATE (NEW FOR MONTH ANALYSIS) =====
    if 'Date_Post' in df.columns:
        # Convert Date_Post to datetime objects, handling dd-mm-yyyy format
        df['Date_Post'] = pd.to_datetime(df['Date_Post'], dayfirst=True, errors='coerce')
        
    return df, initial_count, cleaned_count, pre_test_count_initial, post_test_count_initial

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

# ===== TAB 9: MONTH ANALYSIS (MODIFIED) =====
def tab9_month_analysis(df):
    """
    Generates Month-wise Analysis including:
    1. Bar graph of Number of SESSIONS per month (Chronological)
    2. Line graph of Pre vs Post score % per month (Sorted by Pre-test score)
    
    Logic for Sessions: Unique combination of (School, Class, Content Id, Date)
    """
    st.header("Month-wise Performance Analysis")

    # Filter out rows with missing dates (ensure Date_Post is valid)
    df_month = df.dropna(subset=['Date_Post']).copy()

    if df_month.empty:
        st.warning("No data available with valid 'Date_Post'. Please ensure the excel has 'Date_Post' in dd-mm-yyyy format.")
        return

    # Extract Month information
    # Month_Sort (Period) for chronological sorting
    df_month['Month_Sort'] = df_month['Date_Post'].dt.to_period('M')
    # Month_Display (String) for charts
    df_month['Month_Display'] = df_month['Date_Post'].dt.strftime('%b %Y')

    # ==============================================================================
    # 1. Bar Graph: Number of Sessions per Month (MODIFIED LOGIC)
    # Logic: Group by School, Class, Content Id, Date to find unique sessions first
    # ==============================================================================
    
    # Identify unique sessions
    # We define a session as unique combination of School, Class, Content, and Date
    unique_sessions = df_month.drop_duplicates(subset=['School Name', 'Class', 'Content Id', 'Date_Post'])
    
    # Group by Month_Sort using the UNIQUE SESSIONS dataframe
    sessions_per_month = unique_sessions.groupby('Month_Sort').agg(
        Num_Sessions=('Date_Post', 'count'), # Counting rows here counts unique sessions
        Month_Display=('Month_Display', 'first')
    ).reset_index()

    # Sort strictly chronologically
    sessions_per_month = sessions_per_month.sort_values('Month_Sort')

    st.subheader("📊 Number of Sessions per Month")
    st.markdown("*(A 'Session' is defined as a unique combination of School, Class, Content, and Date)*")

    fig_bar = px.bar(
        sessions_per_month,
        x='Month_Display',
        y='Num_Sessions',
        text='Num_Sessions',
        color_discrete_sequence=['#9b59b6']
    )

    fig_bar.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Sessions",
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(gridcolor='#404040'),
        xaxis=dict(gridcolor='#404040')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ==============================================================================
    # 2. Line Graph: Pre vs Post Score % (Sorted by Pre-Test Score)
    # This remains based on student averages to accurately reflect performance
    # ==============================================================================
    
    # Calculate scores per month
    scores_per_month = df_month.groupby('Month_Display').agg(
        Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
        Avg_Post_Score_Raw=('Post_Score', 'mean')
    ).reset_index()

    # Calculate percentages
    scores_per_month['Avg Pre Score %'] = (scores_per_month['Avg_Pre_Score_Raw'] / 5) * 100
    scores_per_month['Avg Post Score %'] = (scores_per_month['Avg_Post_Score_Raw'] / 5) * 100

    # Sort by Pre Score Ascending (as requested)
    scores_per_month = scores_per_month.sort_values('Avg Pre Score %', ascending=True)

    st.subheader("📈 Pre vs Post Score % per Month (Sorted by Pre-Test Score)")

    fig_line = go.Figure()

    # Pre-Session Trace
    fig_line.add_trace(go.Scatter(
        x=scores_per_month['Month_Display'],
        y=scores_per_month['Avg Pre Score %'],
        mode='lines+markers+text',
        name='Pre-Session Average',
        line=dict(color='#2ecc71', width=3, dash='dot'),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in scores_per_month['Avg Pre Score %']],
        textposition='top center',
        textfont=dict(color='#2ecc71')
    ))

    # Post-Session Trace
    fig_line.add_trace(go.Scatter(
        x=scores_per_month['Month_Display'],
        y=scores_per_month['Avg Post Score %'],
        mode='lines+markers+text',
        name='Post-Session Average',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=10),
        text=[f"{val:.0f}%" for val in scores_per_month['Avg Post Score %']],
        textposition='top center',
        textfont=dict(color='#e67e22')
    ))

    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Score (%)",
        hovermode='x unified',
        height=500,
        plot_bgcolor='#2b2b2b',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100], gridcolor='#404040'),
        xaxis=dict(type='category', gridcolor='#404040') # Treat X axis as category to maintain score-based sort order
    )

    st.plotly_chart(fig_line, use_container_width=True)


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
                
            # UPDATED: Unpack the two new returned variables
            df, initial_count, cleaned_count, pre_test_count_initial, post_test_count_initial = clean_and_process_data(raw_df)
            
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
            
            # UPDATED: The column layout is changed from 4 to 6 columns to fit the new metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Initial Records", initial_count)
            # NEW METRIC 1: Initial Pre-Tests
            with col2:
                st.metric("Initial Pre-Tests", pre_test_count_initial)
            # NEW METRIC 2: Initial Post-Tests
            with col3:
                st.metric("Initial Post-Tests", post_test_count_initial)
            with col4:
                st.metric("Records Removed", records_removed)
            with col5:
                st.metric("Final Records", cleaned_count)
            with col6:
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
        # Ensure Date_Post is converted to string for grouping if it's datetime, or just use as is (pandas handles it)
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
    
    # ===== TABS FOR DIFFERENT ANALYSES (UPDATED WITH MONTH ANALYSIS) =====
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📍 Region Analysis", "👤 Instructor Analysis", "📚 Grade Analysis", "📊 Program Type Analysis", "👥 Student Participation", "🏫 School Analysis", "💰 Donor Analysis", "🔬 Subject Analysis", "📅 Month Analysis"])

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
        
        # Sort in ascending order of PRE Score %
        region_stats = region_stats.sort_values('Pre_Score_Pct', ascending=True)

        # 1. Performance Comparison (Line Chart)
        st.subheader("📈 Region Performance Comparison (Pre vs. Post)")

        fig = go.Figure()
        
        # Pre-Session Trace
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in region_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#2ecc71')
        ))
        
        # Post-Session Trace
        fig.add_trace(go.Scatter(
            x=region_stats['Region'],
            y=region_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f1c40f', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in region_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#f1c40f')
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

        # 2. Region-wise Participation (Bar Chart)
        st.markdown("---")
        st.subheader("Student Participation by Region")
        
        # Ensure region_stats is sorted by region name for the bar chart
        region_stats_participation = region_stats.sort_values('Region')

        fig_part = go.Figure()
        fig_part.add_trace(go.Bar(
            x=region_stats_participation['Region'],
            y=region_stats_participation['Student Id'],
            marker_color='#3498db',
            text=region_stats_participation['Student Id'],
            textposition='outside'
        ))

        fig_part.update_layout(
            title='Total Assessments Conducted by Region',
            xaxis_title='Region',
            yaxis_title='Total Assessments',
            height=400,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040')
        )
        st.plotly_chart(fig_part, use_container_width=True)
        
        # 3. Detailed Table
        st.markdown("---")
        st.subheader("📋 Detailed Region Statistics")

        # Select, rename, and format columns for display
        region_display = region_stats[['Region', 'Student Id', 'Pre_Score_Pct', 'Post_Score_Pct', 'Improvement']].copy()
        region_display.columns = ['Region', 'Total Assessments', 'Avg Pre Score %', 'Avg Post Score %', 'Improvement %']
        region_display['Avg Pre Score %'] = region_display['Avg Pre Score %'].apply(lambda x: f"{x:.1f}%")
        region_display['Avg Post Score %'] = region_display['Avg Post Score %'].apply(lambda x: f"{x:.1f}%")
        region_display['Improvement %'] = region_display['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(region_display, hide_index=True, use_container_width=True)
        
        # 4. Program Type Analysis per Region (Modified for Multi-Select)
        st.markdown("---")
        
        # Calculate scores by Region and Program Type
        program_region_stats = filtered_df.groupby(['Region', 'Program Type']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100

        st.subheader("Region Analysis by Program Type (Multi-Select)")

        # Change selectbox to multiselect
        unique_programs = sorted(filtered_df['Program Type'].unique())
        selected_program_types = st.multiselect("Select Program Type(s) for Detailed View", unique_programs, default=unique_programs, key='tab1_program_select')

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

    # ===== TAB 2: INSTRUCTOR ANALYSIS (UNCHANGED) =====
    with tab2:
        st.header("Instructor-wise Performance and Participation Analysis")
        
        # Check for required columns
        required_cols = ['Instructor Name', 'Instructor Login Id', 'Region']
        if not all(col in filtered_df.columns for col in required_cols):
            st.error(f"❌ Columns {required_cols} not found in the filtered data. Cannot perform Instructor Analysis.")
            st.stop()
            
        # Aggregate data by Instructor Name
        instructor_stats = filtered_df.groupby('Instructor Name').agg(
            Num_Students=('Student Id', 'count'), # Total assessments
            Avg_Pre_Score=('Pre_Score', 'mean'),
            Avg_Post_Score=('Post_Score', 'mean')
        ).reset_index()

        # Calculate percentages and improvement
        instructor_stats['Pre_Score_Pct'] = (instructor_stats['Avg_Pre_Score'] / 5) * 100
        instructor_stats['Post_Score_Pct'] = (instructor_stats['Avg_Post_Score'] / 5) * 100
        instructor_stats['Improvement'] = instructor_stats['Post_Score_Pct'] - instructor_stats['Pre_Score_Pct']
        
        # Filter out instructors with less than 10 assessments
        min_assessments = st.slider("Minimum Assessments for Performance Metrics", min_value=1, max_value=50, value=10)
        instructor_stats_filtered = instructor_stats[instructor_stats['Num_Students'] >= min_assessments]
        
        st.subheader(f"Top 10 Instructors (Min. {min_assessments} Assessments)")
        
        if instructor_stats_filtered.empty:
            st.info(f"No instructors meet the minimum assessment requirement of {min_assessments}.")
        else:
            col1, col2 = st.columns(2)
            
            # Prepare table data (keep floats for sorting, then format for display)
            instructor_stats_for_table = instructor_stats_filtered.copy()
            instructor_stats_for_table.columns = ['Instructor Name', 'Total Assessments', 'Avg Pre Score (Raw)', 'Avg Post Score (Raw)', 'Avg Pre %', 'Avg Post %', 'Improvement']

            with col1:
                st.subheader("🥇 Best Performance (Post-Score)")
                # Sort by Post Score %
                top_perf = instructor_stats_for_table.nlargest(10, 'Avg Post %')[['Instructor Name', 'Avg Post %', 'Total Assessments']]
                top_perf.columns = ['Instructor', 'Avg Post %', 'Assessments']
                # Format score column for display
                top_perf['Avg Post %'] = top_perf['Avg Post %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top_perf, hide_index=True, use_container_width=True)

            with col2:
                st.subheader("📈 Best Adaptation (Improvement)")
                # Sort by Improvement
                best_adapt = instructor_stats_for_table.nlargest(10, 'Improvement')[['Instructor Name', 'Improvement', 'Total Assessments']]
                best_adapt.columns = ['Instructor', 'Improvement %', 'Assessments']
                # Format score column for display
                best_adapt['Improvement %'] = best_adapt['Improvement %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(best_adapt, hide_index=True, use_container_width=True)
                
        # All Instructors Assessment Count
        st.markdown("---")
        st.subheader("📋 Complete Instructor List - Assessment Count")
        
        # FIX APPLIED: Correctly calculate sessions using a unique key
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
        instructors_per_region.columns = ['Region', 'Unique Instructors']
        instructors_per_region = instructors_per_region.sort_values('Unique Instructors', ascending=False)
        
        # Plotting
        fig_inst_region = px.bar(
            instructors_per_region,
            x='Region',
            y='Unique Instructors',
            text='Unique Instructors',
            color_discrete_sequence=px.colors.qualitative.Dark
        )
        fig_inst_region.update_layout(
            title='Unique Instructor Count by Region',
            xaxis_title='Region',
            yaxis_title='Unique Instructors',
            height=500,
            plot_bgcolor='#2b2b2b',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            yaxis=dict(gridcolor='#404040'),
            xaxis=dict(type='category', gridcolor='#404040')
        )
        st.plotly_chart(fig_inst_region, use_container_width=True)


    # ===== TAB 3: GRADE ANALYSIS (UNCHANGED) =====
    with tab3:
        st.header("Grade-wise Performance Analysis")
        
        # Aggregate data by Parent Class (Grade)
        grade_stats = filtered_df.groupby('Parent_Class').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count' # Total assessments
        }).reset_index()

        # Calculate percentages and improvement
        grade_stats['Pre_Score_Pct'] = (grade_stats['Pre_Score'] / 5) * 100
        grade_stats['Post_Score_Pct'] = (grade_stats['Post_Score'] / 5) * 100
        grade_stats['Improvement'] = grade_stats['Post_Score_Pct'] - grade_stats['Pre_Score_Pct']
        
        # Convert Parent_Class to numeric for sorting, then back to string for chart display
        grade_stats['Parent_Class'] = pd.to_numeric(grade_stats['Parent_Class'], errors='coerce')
        # Sort in ascending order of PRE Score %
        grade_stats = grade_stats.sort_values('Pre_Score_Pct', ascending=True)
        # Convert back to string for plotting
        grade_stats['Parent_Class'] = grade_stats['Parent_Class'].astype(str)

        # 1. Performance Comparison (Line Chart)
        st.subheader("📈 Grade Performance Comparison (Pre vs. Post)")

        fig = go.Figure()
        
        # Pre-Session Trace
        fig.add_trace(go.Scatter(
            x=grade_stats['Parent_Class'],
            y=grade_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in grade_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#2ecc71')
        ))
        
        # Post-Session Trace
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

        # 2. Grade-wise participation (Bar Chart)
        st.markdown("---")
        st.subheader("Student Participation by Grade")
        
        # Ensure grade_stats is sorted by grade number for the bar chart
        grade_stats_participation = grade_stats.copy()
        grade_stats_participation['Parent_Class'] = pd.to_numeric(grade_stats_participation['Parent_Class'])
        grade_stats_participation = grade_stats_participation.sort_values('Parent_Class')
        grade_stats_participation['Parent_Class'] = grade_stats_participation['Parent_Class'].astype(str)
        
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
        
        # 3. Detailed table
        st.subheader("Detailed Grade Statistics")
        grade_display = grade_stats[['Parent_Class', 'Student Id', 'Pre_Score_Pct', 'Post_Score_Pct', 'Improvement']].copy()
        grade_display.columns = ['Grade', 'Total Assessments', 'Avg Pre Score %', 'Avg Post Score %', 'Improvement %']
        grade_display['Avg Pre Score %'] = grade_display['Avg Pre Score %'].apply(lambda x: f"{x:.1f}%")
        grade_display['Avg Post Score %'] = grade_display['Avg Post Score %'].apply(lambda x: f"{x:.1f}%")
        grade_display['Improvement %'] = grade_display['Improvement %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(grade_display, hide_index=True, use_container_width=True)


    # ===== TAB 4: PROGRAM TYPE ANALYSIS (UNCHANGED) =====
    with tab4:
        st.header("Program Type Analysis")
        
        # Aggregate data by Program Type
        program_stats = filtered_df.groupby('Program Type').agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean',
            'Student Id': 'count' # Total assessments
        }).reset_index()

        # Calculate percentages and improvement
        program_stats['Pre_Score_Pct'] = (program_stats['Pre_Score'] / 5) * 100
        program_stats['Post_Score_Pct'] = (program_stats['Post_Score'] / 5) * 100
        program_stats['Improvement'] = program_stats['Post_Score_Pct'] - program_stats['Pre_Score_Pct']
        
        # Sort in ascending order of PRE Score %
        program_stats = program_stats.sort_values('Pre_Score_Pct', ascending=True)

        # 1. Performance Comparison (Line Chart)
        st.subheader("📈 Program Type Performance Comparison (Pre vs. Post)")

        fig = go.Figure()
        
        # Pre-Session Trace
        fig.add_trace(go.Scatter(
            x=program_stats['Program Type'],
            y=program_stats['Pre_Score_Pct'],
            mode='lines+markers+text',
            name='Pre-Session',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in program_stats['Pre_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#2ecc71')
        ))
        
        # Post-Session Trace
        fig.add_trace(go.Scatter(
            x=program_stats['Program Type'],
            y=program_stats['Post_Score_Pct'],
            mode='lines+markers+text',
            name='Post-Session',
            line=dict(color='#f1c40f', width=3),
            marker=dict(size=12),
            text=[f"{val:.0f}%" for val in program_stats['Post_Score_Pct']],
            textposition='top center',
            textfont=dict(size=14, color='#f1c40f')
        ))
        
        fig.update_layout(
            title='Program Type Performance Comparison (Ascending by Pre Score)',
            xaxis_title='Program Type',
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

        # 2. Program Type-wise participation (Bar Chart)
        st.markdown("---")
        st.subheader("Student Participation by Program Type")
        
        # Aggregate unique students per program type
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

        # 3. Detailed table
        st.markdown("---")
        st.subheader("📋 Detailed Program Type Statistics")

        # Select, rename, and format columns for display
        program_display = program_stats[['Program Type', 'Student Id', 'Pre_Score_Pct', 'Post_Score_Pct', 'Improvement']].copy()
        program_display.columns = ['Program Type', 'Total Assessments', 'Avg Pre Score %', 'Avg Post Score %', 'Improvement %']
        program_display['Avg Pre Score %'] = program_display['Avg Pre Score %'].apply(lambda x: f"{x:.1f}%")
        program_display['Avg Post Score %'] = program_display['Avg Post Score %'].apply(lambda x: f"{x:.1f}%")
        program_display['Improvement %'] = program_display['Improvement %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(program_display, hide_index=True, use_container_width=True)
        
        # 4. Program vs Region Comparison (Multi-Select)
        st.markdown("---")
        st.subheader("🗺️ Program Performance Comparison by Region")

        # Calculate scores by Program Type and Region
        program_region_stats = filtered_df.groupby(['Program Type', 'Region']).agg({
            'Pre_Score': 'mean',
            'Post_Score': 'mean'
        }).reset_index()
        program_region_stats['Pre_Score_Pct'] = (program_region_stats['Pre_Score'] / 5) * 100
        program_region_stats['Post_Score_Pct'] = (program_region_stats['Post_Score'] / 5) * 100
        
        unique_regions_in_data = sorted(filtered_df['Region'].unique())
        selected_regions_for_prog = st.multiselect("Select Region(s) to compare Programs", unique_regions_in_data, default=unique_regions_in_data, key='prog_region_viz_select_multi')

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

    # ===== TAB 5: STUDENT PARTICIPATION (UNCHANGED) =====
    with tab5:
        st.header("Student Participation Analysis")
        
        # 1. Total Unique Students
        total_unique_students = filtered_df['Student Id'].nunique()
        st.metric("Total Unique Students Assessed", total_unique_students)
        st.markdown("---")
        
        # 2. Student Activity (Tests per student)
        st.subheader("Assessment Frequency per Student")
        
        if not filtered_df.empty:
            # Re-calculate student activity for the filtered data (already done in main, but re-run for consistency)
            unique_student_tests = filtered_df.groupby(
                ['Student Id', 'Content Id', 'Class', 'School Name', 'Date_Post']
            ).size().reset_index(name='count')
            student_activity = unique_student_tests.groupby('Student Id').size().reset_index(name='Tests_Taken')
            student_activity = student_activity.sort_values('Tests_Taken', ascending=False)
            
            # Calculate frequency distribution
            test_counts = student_activity.groupby('Tests_Taken').size().reset_index(name='Num Students')
            test_counts['Percentage'] = (test_counts['Num Students'] / test_counts['Num Students'].sum()) * 100
            test_counts = test_counts.sort_values('Tests_Taken', ascending=False)
            
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Tests/Student", f"{student_activity['Tests_Taken'].mean():.1f}")
            with col2:
                st.metric("Max Tests/Student", f"{student_activity['Tests_Taken'].max()}")
            with col3:
                st.metric("Min Tests/Student", f"{student_activity['Tests_Taken'].min()}")
                
            st.markdown("---")
            
            # Plotting the distribution
            st.subheader("Distribution of Tests Taken by Students")
            
            fig_dist = px.bar(
                test_counts,
                x='Tests_Taken',
                y='Num Students',
                text='Num Students',
                color_discrete_sequence=['#3498db']
            )

            fig_dist.update_layout(
                title='Number of Students by Assessment Count',
                xaxis_title='Number of Unique Assessments Taken',
                yaxis_title='Number of Students',
                height=500,
                plot_bgcolor='#2b2b2b',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                yaxis=dict(gridcolor='#404040'),
                xaxis=dict(dtick=1, gridcolor='#404040') # Force integer ticks on X-axis
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Display frequency table
            st.subheader("Detailed Frequency Table")
            test_counts_display = test_counts.copy()
            test_counts_display.columns = ['Tests Taken', 'Number of Students', 'Percentage']
            test_counts_display['Percentage'] = test_counts_display['Percentage'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(test_counts_display, hide_index=True, use_container_width=True)
            
            # Download button for detailed activity
            student_activity_csv = student_activity.to_csv(index=False)
            st.download_button(
                "📥 Download Student Assessment Frequency (CSV)",
                student_activity_csv,
                "student_assessment_frequency.csv",
                "text/csv"
            )

        else:
            st.info("No student data available after filters.")


    # ===== TAB 6: SCHOOL ANALYSIS (UNCHANGED) =====
    with tab6:
        st.header("School-wise Performance Analysis")
        try:
            # Aggregate data by School
            school_stats = filtered_df.groupby(['School Name', 'UDISE']).agg(
                Total_Students=('Student Id', 'nunique'),
                Unique_Assessments=('Student Id', 'count'),
                Total_Instructors=('Instructor Name', 'nunique'),
                Region=('Region', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]) # Most common region
            ).reset_index()

            # Sort by total students
            school_stats = school_stats.sort_values('Total_Students', ascending=False)
            
            # Rename columns
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
                # Filter both on School Name (case-insensitive) and UDISE
                school_stats_display = school_stats[
                    school_stats['School Name'].astype(str).str.contains(search_school, case=False) |
                    school_stats['UDISE'].astype(str).str.contains(search_school, case=False)
                ].copy()
            else:
                school_stats_display = school_stats.copy()
            
            st.dataframe(school_stats_display, hide_index=True, use_container_width=True)
            
            # Download Button
            school_csv = school_stats.to_csv(index=False)
            st.download_button(
                "📥 Download School Analysis Data (CSV)",
                school_csv,
                "school_analysis.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generating school analysis: {str(e)}")


    # ===== TAB 7: DONOR ANALYSIS (UNCHANGED) =====
    with tab7:
        st.header("Donor-wise Performance Analysis")
        
        # 1. Donor Selection for Detailed View
        all_donors = sorted(filtered_df['Donor'].unique().tolist())
        selected_donor = st.selectbox("Select Donor for Detailed Regional Breakdown", ['ALL DONORS (Summary)'] + all_donors)

        st.markdown("---")

        # --- LOGIC FOR ALL DONORS (Summary View) ---
        # Calculate summary statistics for ALL donors present in the filtered view
        donor_stats = filtered_df.groupby('Donor').agg(
            Num_Schools=('UDISE', 'nunique'),
            Num_Students=('Student Id', 'nunique'),
            Num_Assessments=('Student Id', 'count'),
            Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
            Avg_Post_Score_Raw=('Post_Score', 'mean')
        ).reset_index()

        # 2. Calculate percentages (vectorized)
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

        st.subheader("📋 Donor-wise Summary (Responds to Global Filters)") # Show Raw Scores and Percentages
        st.dataframe(display_donor_stats, hide_index=True, use_container_width=True)
        
        st.markdown("---")

        # --- LOGIC FOR INDIVIDUAL DONOR (Specific Metrics) ---

        if selected_donor == 'ALL DONORS (Summary)':
            donor_filtered_df = filtered_df.copy() # Use the full filtered set
        else:
            donor_filtered_df = filtered_df[filtered_df['Donor'] == selected_donor].copy()
        
        st.subheader(f"Key Metrics for **{selected_donor}**")

        if donor_filtered_df.empty:
            st.info("No data available for this selection.")
            # Ensure the rest of the tab doesn't error out
            donor_specific_stats = {
                'Avg Pre Score %': 0.0,
                'Avg Post Score %': 0.0,
                'Improvement %': 0.0,
                'Total Assessments': 0,
                'Total Students': 0,
                'Total Schools': 0
            }
            donor_region_stats = pd.DataFrame()
            
        else:
            # Calculate Metrics for the Selected Donor/All (uses the now correctly filtered donor_filtered_df)
            donor_specific_stats = {
                'Avg Pre Score %': (donor_filtered_df['Pre_Score'].mean() / 5) * 100,
                'Avg Post Score %': (donor_filtered_df['Post_Score'].mean() / 5) * 100,
                'Improvement %': (donor_filtered_df['Post_Score'].mean() - donor_filtered_df['Pre_Score'].mean()) / 5 * 100,
                'Total Assessments': len(donor_filtered_df),
                'Total Students': donor_filtered_df['Student Id'].nunique(),
                'Total Schools': donor_filtered_df['UDISE'].nunique()
            }
            
            # Display Key Metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Avg Pre Score", f"{donor_specific_stats['Avg Pre Score %']:.1f}%")
            with col2:
                st.metric("Avg Post Score", f"{donor_specific_stats['Avg Post Score %']:.1f}%")
            with col3:
                st.metric("Improvement", f"{donor_specific_stats['Improvement %']:.1f}%", delta=f"{donor_specific_stats['Improvement %']:.1f}%")
            with col4:
                st.metric("Total Assessments", donor_specific_stats['Total Assessments'])
            with col5:
                st.metric("Total Students", donor_specific_stats['Total Students'])
            with col6:
                st.metric("Total Schools", donor_specific_stats['Total Schools'])

            # --- REGIONAL BREAKDOWN (Visualization) ---
            st.markdown("---")
            st.subheader(f"Regional Performance Breakdown for **{selected_donor}**")

            # Group by Region for the selected donor/all
            donor_region_stats = donor_filtered_df.groupby('Region').agg(
                Num_Schools=('UDISE', 'nunique'),
                Num_Students=('Student Id', 'nunique'),
                Avg_Pre_Score_Raw=('Pre_Score', 'mean'),
                Avg_Post_Score_Raw=('Post_Score', 'mean')
            ).reset_index()

            donor_region_stats['Avg Pre Score %'] = (donor_region_stats['Avg_Pre_Score_Raw'] / 5) * 100
            donor_region_stats['Avg Post Score %'] = (donor_region_stats['Avg_Post_Score_Raw'] / 5) * 100
            donor_region_stats['Improvement %'] = donor_region_stats['Avg Post Score %'] - donor_region_stats['Avg Pre Score %']
            
            # Sort by Improvement %
            donor_region_stats = donor_region_stats.sort_values('Improvement %', ascending=False)
            
            # Plotting Regional Breakdown
            fig_donor_region = go.Figure()

            # Improvement Trace (Bar Chart)
            fig_donor_region.add_trace(go.Bar(
                x=donor_region_stats['Region'],
                y=donor_region_stats['Improvement %'],
                name='Improvement',
                marker_color=['#2ecc71' if i > 0 else '#e74c3c' for i in donor_region_stats['Improvement %']],
                text=[f"{val:.1f}%" for val in donor_region_stats['Improvement %']],
                textposition='outside'
            ))

            fig_donor_region.update_layout(
                title=f'Regional Improvement for {selected_donor}',
                xaxis_title='Region',
                yaxis_title='Improvement (%)',
                height=500,
                plot_bgcolor='#2b2b2b',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                yaxis=dict(gridcolor='#404040'),
                xaxis=dict(type='category', gridcolor='#404040')
            )
            st.plotly_chart(fig_donor_region, use_container_width=True)

            # --- REGIONAL BREAKDOWN (Table) ---
            st.markdown("---")
            st.subheader(f"📋 Regional Detailed Statistics for **{selected_donor}**")
            
            # Format columns for display
            display_donor_region = donor_region_stats.copy()
            display_donor_region['Avg Pre Score %'] = display_donor_region['Avg Pre Score %'].apply(lambda x: f"{x:.1f}%")
            display_donor_region['Avg Post Score %'] = display_donor_region['Avg Post Score %'].apply(lambda x: f"{x:.1f}%")
            display_donor_region['Improvement %'] = display_donor_region['Improvement %'].apply(lambda x: f"{x:.1f}%")

            # Select and rename columns
            display_donor_region = display_donor_region[['Region', 'Num_Schools', 'Num_Students', 'Avg Pre Score %', 'Avg Post Score %', 'Improvement %']].copy()
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

    # ===== TAB 9: MONTH ANALYSIS (UNCHANGED) =====
    with tab9:
        if not filtered_df.empty:
            tab9_month_analysis(filtered_df)
        else:
            st.info("No data to display after applying filters.")
            
    # ===== GLOBAL DOWNLOAD SECTION (UNCHANGED) =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Data Download")

    if subject_stats is not None and not subject_stats.empty:
        # Prepare an overall summary table combining key metrics
        
        # Calculate overall key performance indicators (KPIs)
        total_students_assessed = filtered_df['Student Id'].nunique()
        total_assessments = len(filtered_df)
        
        avg_pre = (filtered_df['Pre_Score'].mean() / 5) * 100
        avg_post = (filtered_df['Post_Score'].mean() / 5) * 100
        improvement = avg_post - avg_pre
        
        # Calculate instructor/school/region counts from the full filtered_df
        unique_regions = filtered_df['Region'].nunique()
        unique_instructors = filtered_df['Instructor Name'].nunique()
        unique_schools = filtered_df['School Name'].nunique()
        
        overall_summary = pd.DataFrame({
            'Metric': [
                'Total Unique Students Assessed', 
                'Total Assessments Conducted', 
                'Avg Pre Score (%)', 
                'Avg Post Score (%)', 
                'Overall Improvement (%)',
                'Unique Regions in View',
                'Unique Instructors in View',
                'Unique Schools in View'
            ],
            'Value': [
                total_students_assessed, 
                total_assessments, 
                f"{avg_pre:.1f}%", 
                f"{avg_post:.1f}%", 
                f"{improvement:.1f}%",
                unique_regions,
                unique_instructors,
                unique_schools
            ]
        })
        
        # Provide a general summary download (CSV)
        overall_summary_csv = overall_summary.to_csv(index=False)
        st.sidebar.download_button(
            "📥 Download Overall Summary (CSV)",
            overall_summary_csv,
            "overall_summary.csv",
            "text/csv"
        )
    
        # Provide the cleaned, filtered data for custom analysis
        filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=filtered_csv,
            file_name="filtered_student_data.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload an Excel file to begin the analysis.")