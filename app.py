import streamlit as st
import pandas as pd
from rapidfuzz import process
import ast
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="ICD-10 Diagnosis Mapper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def prepare_icd_data_from_txt():
    """
    Prepare ICD-10 data from the text file and save as clean CSV
    """
    try:
        with open('icd10cm_order_2024.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        records = []
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Split into parts
                # Format: line_number ICD_code level description [duplicate_description]
                parts = line.split()
                if len(parts) >= 4:
                    line_num = parts[0]    # e.g., "00001"
                    icd_code = parts[1]    # e.g., "A00"
                    level = parts[2]       # e.g., "0"
                    
                    # Get the description part
                    # The description appears to be duplicated, so we need to extract just one copy
                    desc_parts = parts[3:]
                    
                    # Method: Take first half (assuming duplication)
                    mid_point = len(desc_parts) // 2
                    if mid_point > 0:
                        description = ' '.join(desc_parts[:mid_point]).strip()
                    else:
                        description = ' '.join(desc_parts).strip()
                    
                    # Clean up the description (remove extra spaces)
                    description = ' '.join(description.split())
                    
                    # Validate we have meaningful data
                    if icd_code and description and len(description) > 3:
                        records.append({
                            'ICD_Code': icd_code,
                            'ICD_Description': description
                        })
                        
            except Exception as e:
                continue
        
        # Create DataFrame and remove duplicates
        icd_df = pd.DataFrame(records)
        icd_df = icd_df.drop_duplicates(subset=['ICD_Code'], keep='first')
        
        # Save as CSV for future use
        icd_df.to_csv('icd10_codes_clean.csv', index=False)
        
        return icd_df
        
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

@st.cache_data
def load_data():
    """Load the ICD-10 codes and mapped diagnoses data"""
    try:
        # Load your exported mapped diagnoses
        if os.path.exists("mapped_diagnoses_with_icd.csv"):
            mapped_df = pd.read_csv("mapped_diagnoses_with_icd.csv")
        else:
            st.error("❌ mapped_diagnoses_with_icd.csv not found")
            st.info("Please upload your mapped diagnoses file using the file uploader below")
            return None, None
        
        # Try to load cleaned ICD data first, then fall back to original file
        icd_df = None
        
        # Check if clean CSV exists
        if os.path.exists("icd10_codes_clean.csv"):
            try:
                icd_df = pd.read_csv("icd10_codes_clean.csv")
                st.success("✅ Loaded clean ICD-10 data")
            except:
                pass
        
        # If clean CSV doesn't exist or failed to load, process the original file
        if icd_df is None:
            if os.path.exists('icd10cm_order_2024.txt'):
                st.info("Processing ICD-10 data from original file...")
                with st.spinner("Parsing ICD-10 codes... This may take a moment."):
                    icd_df = prepare_icd_data_from_txt()
                
                if icd_df is not None and len(icd_df) > 0:
                    st.success(f"✅ Processed and saved {len(icd_df)} ICD codes")
                else:
                    st.error("Failed to process ICD-10 file")
                    return None, None
            else:
                # If no ICD file, try to use codes from mapped data
                st.warning("⚠️ ICD-10 source file not found")
                if 'ICD_Code' in mapped_df.columns and 'ICD_Description' in mapped_df.columns:
                    st.info("Using ICD codes from mapped diagnoses data...")
                    icd_df = mapped_df[['ICD_Code', 'ICD_Description']].dropna().drop_duplicates()
                    if len(icd_df) > 0:
                        st.success(f"✅ Using {len(icd_df)} ICD codes from mapped data")
                    else:
                        st.error("No valid ICD codes found in mapped data")
                        return None, None
                else:
                    st.error("❌ No ICD-10 data source available")
                    st.info("Please upload either:")
                    st.info("1. icd10cm_order_2024.txt - Original ICD-10 codes file")
                    st.info("2. Mapped diagnoses CSV with ICD_Code and ICD_Description columns")
                    return None, None
        
        # Final data validation
        if icd_df is not None and len(icd_df) > 0:
            # Ensure required columns exist
            if 'ICD_Code' not in icd_df.columns or 'ICD_Description' not in icd_df.columns:
                st.error("❌ ICD data missing required columns: ICD_Code, ICD_Description")
                return None, None
            
            icd_df = icd_df.dropna()
            icd_df['ICD_Code'] = icd_df['ICD_Code'].astype(str)
            icd_df['ICD_Description'] = icd_df['ICD_Description'].astype(str)
            
            # Remove any rows where code or description is empty
            icd_df = icd_df[(icd_df['ICD_Code'] != '') & (icd_df['ICD_Description'] != '')]
            
            # Remove duplicates
            icd_df = icd_df.drop_duplicates(subset=['ICD_Code'])
        else:
            st.error("Could not load ICD-10 data")
            return None, None
            
        return mapped_df, icd_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def map_to_icd_with_alternatives(diagnosis, icd_df, threshold=80, top_n=5):
    """Map diagnosis to ICD-10 codes with fuzzy matching"""
    choices = icd_df['ICD_Description'].tolist()
    top_matches = process.extract(diagnosis, choices, limit=top_n)
    
    best_match, best_score, _ = top_matches[0]
    
    if best_score >= threshold:
        icd_row = icd_df[icd_df['ICD_Description'] == best_match].iloc[0]
        return {
            'Original_Diagnosis': diagnosis,
            'ICD_Code': icd_row['ICD_Code'],
            'ICD_Description': icd_row['ICD_Description'],
            'Match_Score': best_score,
            'Suggested_Alternatives': None,
            'Justification': f"Fuzzy matched to: '{best_match}' with score {best_score}"
        }
    else:
        alt_suggestions = []
        for match_text, score, _ in top_matches:
            row = icd_df[icd_df['ICD_Description'] == match_text].iloc[0]
            alt_suggestions.append({
                'ICD_Code': row['ICD_Code'],
                'ICD_Description': row['ICD_Description'],
                'Score': score
            })
        return {
            'Original_Diagnosis': diagnosis,
            'ICD_Code': None,
            'ICD_Description': None,
            'Match_Score': best_score,
            'Suggested_Alternatives': alt_suggestions,
            'Justification': f"No confident match (top score {best_score}). Suggested alternatives provided."
        }

def file_upload_section():
    """File upload section for deployment"""
    st.header("📁 File Upload")
    st.info("Upload your data files to get started:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mapped Diagnoses CSV")
        mapped_file = st.file_uploader(
            "Upload mapped_diagnoses_with_icd.csv",
            type=['csv'],
            key="mapped_file"
        )
        
        if mapped_file:
            try:
                df = pd.read_csv(mapped_file)
                st.success(f"✅ Loaded {len(df)} mapped diagnoses")
                st.write("Preview:")
                st.dataframe(df.head(), use_container_width=True)
                
                # Save the file for processing
                df.to_csv("mapped_diagnoses_with_icd.csv", index=False)
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.subheader("ICD-10 Codes File")
        icd_file = st.file_uploader(
            "Upload icd10cm_order_2024.txt",
            type=['txt'],
            key="icd_file"
        )
        
        if icd_file:
            try:
                # Save the uploaded file
                with open("icd10cm_order_2024.txt", "wb") as f:
                    f.write(icd_file.getbuffer())
                st.success("✅ ICD-10 file uploaded successfully")
                
                # Process and show preview
                with st.spinner("Processing file..."):
                    result = prepare_icd_data_from_txt()
                    if result is not None:
                        st.success(f"✅ Processed {len(result)} ICD codes")
                        st.write("Preview:")
                        st.dataframe(result.head(), use_container_width=True)
                    else:
                        st.error("Failed to process ICD file")
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")

def data_preparation_section():
    """Section for manual data preparation if needed"""
    st.header("🔧 Data Preparation")
    
    st.write("Use this section if you need to manually process your ICD-10 data file.")
    
    if st.button("🚀 Process ICD-10 File", type="secondary"):
        if os.path.exists('icd10cm_order_2024.txt'):
            with st.spinner("Processing ICD-10 file..."):
                result = prepare_icd_data_from_txt()
                
                if result is not None and len(result) > 0:
                    st.success(f"✅ Successfully processed {len(result)} ICD codes!")
                    st.info("Clean data saved as 'icd10_codes_clean.csv'")
                    
                    # Show sample data
                    st.subheader("Sample of processed data:")
                    st.dataframe(result.head(10))
                    
                    # Data quality info
                    st.subheader("Data Quality Summary:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total ICD Codes", len(result))
                    with col2:
                        avg_desc_len = result['ICD_Description'].str.len().mean()
                        st.metric("Avg Description Length", f"{avg_desc_len:.0f} chars")
                    with col3:
                        categories = result['ICD_Code'].str[0].nunique()
                        st.metric("Categories", categories)
                        
                    # Show categories breakdown
                    st.subheader("Categories Breakdown:")
                    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
                    for cat in categories:
                        cat_codes = result[result['ICD_Code'].str.startswith(cat)]
                        if len(cat_codes) > 0:
                            st.write(f"**{cat}**: {len(cat_codes)} codes (e.g., {cat_codes.iloc[0]['ICD_Code']} - {cat_codes.iloc[0]['ICD_Description'][:50]}...)")
                    
                    st.success("🎉 You can now use the other tabs!")
                    
                else:
                    st.error("❌ Failed to process the file. Please check the file format.")
        else:
            st.error("❌ File 'icd10cm_order_2024.txt' not found. Please upload it first.")

def main():
    st.title("🏥 ICD-10 Diagnosis Mapper")
    st.markdown("---")
    
    # Load data
    mapped_df, icd_df = load_data()
    
    # If data loading failed, show file upload section
    if mapped_df is None or icd_df is None:
        file_upload_section()
        st.stop()
    
    # Sidebar
    st.sidebar.header("Configuration")
    threshold = st.sidebar.slider("Match Threshold", 50, 100, 80, 5)
    top_n = st.sidebar.slider("Number of Alternatives", 3, 10, 5)
    
    # Show data info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Data Info")
        st.metric("ICD Codes", len(icd_df))
        st.metric("Mapped Diagnoses", len(mapped_df))
        
        # Show data prep section in sidebar if needed
        with st.expander("🔧 Data Tools"):
            if st.button("🔄 Reprocess ICD Data"):
                if os.path.exists('icd10cm_order_2024.txt'):
                    with st.spinner("Reprocessing..."):
                        result = prepare_icd_data_from_txt()
                        if result is not None:
                            st.success("✅ Data reprocessed!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to reprocess")
                else:
                    st.error("❌ Source file not found")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Single Query", 
        "📋 Batch Query", 
        "📊 Data Explorer", 
        "📈 Analytics", 
        "🔧 Data Prep"
    ])
    
    with tab1:
        st.header("Single Diagnosis Query")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            diagnosis_input = st.text_input(
                "Enter a diagnosis:",
                placeholder="e.g., diabetes mellitus, hypertension, pneumonia"
            )
            
            if st.button("🔍 Search ICD-10 Code", type="primary"):
                if diagnosis_input:
                    with st.spinner("Searching for ICD-10 match..."):
                        result = map_to_icd_with_alternatives(
                            diagnosis_input, icd_df, threshold, top_n
                        )
                    
                    # Display results
                    if result['ICD_Code']:
                        st.success("✅ Match Found!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("ICD-10 Code", result['ICD_Code'])
                        with col_b:
                            st.metric("Match Score", f"{result['Match_Score']}%")
                        with col_c:
                            st.metric("Confidence", "High" if result['Match_Score'] >= 90 else "Medium")
                        
                        st.subheader("📝 Description")
                        st.info(result['ICD_Description'])
                        
                        st.subheader("🔍 Details")
                        st.write(f"**Original Input:** {result['Original_Diagnosis']}")
                        st.write(f"**Justification:** {result['Justification']}")
                        
                    else:
                        st.warning("⚠️ No confident match found")
                        st.write(f"**Match Score:** {result['Match_Score']}%")
                        
                        if result['Suggested_Alternatives']:
                            st.subheader("💡 Suggested Alternatives")
                            for i, alt in enumerate(result['Suggested_Alternatives']):
                                with st.expander(f"Alternative {i+1}: {alt['ICD_Code']} (Score: {alt['Score']}%)"):
                                    st.write(f"**Description:** {alt['ICD_Description']}")
                else:
                    st.warning("Please enter a diagnosis to search.")
        
        with col2:
            st.subheader("📊 Quick Stats")
            st.metric("Total ICD-10 Codes", len(icd_df))
            st.metric("Mapped Diagnoses", len(mapped_df))
            
            # Success rate
            if 'ICD_Code' in mapped_df.columns:
                successful_matches = len(mapped_df[mapped_df['ICD_Code'].notna()])
                success_rate = (successful_matches / len(mapped_df)) * 100 if len(mapped_df) > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with tab2:
        st.header("Batch Diagnosis Query")
        
        st.write("Upload multiple diagnoses or enter them manually:")
        
        # Text area input
        batch_input = st.text_area(
            "Enter diagnoses (one per line):",
            height=150,
            placeholder="diabetes mellitus\nhypertension\npneumonia\narthritis"
        )
        
        # File upload
        uploaded_file = st.file_uploader("Or upload a CSV file", type=['csv'])
        
        if st.button("🚀 Process Batch", type="primary"):
            diagnoses_list = []
            
            if uploaded_file:
                df_upload = pd.read_csv(uploaded_file)
                if 'diagnosis' in df_upload.columns:
                    diagnoses_list = df_upload['diagnosis'].dropna().tolist()
                else:
                    st.error("CSV file should have a 'diagnosis' column")
                    st.stop()
            elif batch_input:
                diagnoses_list = [diag.strip() for diag in batch_input.split('\n') if diag.strip()]
            
            if diagnoses_list:
                st.write(f"Processing {len(diagnoses_list)} diagnoses...")
                
                results = []
                progress_bar = st.progress(0)
                
                for i, diagnosis in enumerate(diagnoses_list):
                    result = map_to_icd_with_alternatives(diagnosis, icd_df, threshold, top_n)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(diagnoses_list))
                
                # Display results
                results_df = pd.DataFrame(results)
                
                st.subheader("📊 Batch Results Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    successful = len(results_df[results_df['ICD_Code'].notna()])
                    st.metric("Successful Matches", f"{successful}/{len(results_df)}")
                
                with col2:
                    avg_score = results_df['Match_Score'].mean()
                    st.metric("Average Match Score", f"{avg_score:.1f}%")
                
                with col3:
                    high_confidence = len(results_df[results_df['Match_Score'] >= 90])
                    st.metric("High Confidence Matches", high_confidence)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv,
                    "batch_icd_mapping_results.csv",
                    "text/csv"
                )
                
                # Display detailed results
                st.subheader("📋 Detailed Results")
                st.dataframe(
                    results_df[['Original_Diagnosis', 'ICD_Code', 'ICD_Description', 'Match_Score']],
                    use_container_width=True
                )
    
    with tab3:
        st.header("Data Explorer")
        
        # Search in existing mapped data
        st.subheader("🔍 Search Existing Mappings")
        search_term = st.text_input("Search in mapped diagnoses:")
        
        if search_term:
            # Check if required columns exist
            searchable_columns = []
            if 'Original_Diagnosis' in mapped_df.columns:
                searchable_columns.append('Original_Diagnosis')
            if 'ICD_Description' in mapped_df.columns:
                searchable_columns.append('ICD_Description')
            
            if searchable_columns:
                mask = False
                for col in searchable_columns:
                    mask |= mapped_df[col].str.contains(search_term, case=False, na=False)
                
                filtered_df = mapped_df[mask]
                st.write(f"Found {len(filtered_df)} matches:")
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.warning("No searchable columns found in mapped data")
        
        # Browse by ICD code range
        st.subheader("📚 Browse by ICD-10 Code Range")
        code_prefix = st.selectbox(
            "Select ICD-10 category:",
            ["A00-B99: Infectious diseases", "C00-D49: Neoplasms", "E00-E89: Endocrine diseases", 
             "F01-F99: Mental disorders", "G00-G99: Nervous system", "H00-H59: Eye diseases",
             "I00-I99: Circulatory system", "J00-J99: Respiratory system", "K00-K95: Digestive system"]
        )
        
        if code_prefix:
            prefix = code_prefix.split(":")[0].split("-")[0]
            filtered_icd = icd_df[icd_df['ICD_Code'].str.startswith(prefix)]
            st.dataframe(filtered_icd.head(20), use_container_width=True)
    
    with tab4:
        st.header("Analytics Dashboard")
        
        if not mapped_df.empty and 'Match_Score' in mapped_df.columns:
            # Match score distribution
            st.subheader("📊 Match Score Distribution")
            valid_scores = mapped_df[mapped_df['Match_Score'].notna()]
            if len(valid_scores) > 0:
                fig_hist = px.histogram(
                    valid_scores, 
                    x='Match_Score',
                    nbins=20,
                    title="Distribution of Match Scores"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Success rate by score ranges
                st.subheader("📈 Success Rate Analysis")
                score_ranges = pd.cut(mapped_df['Match_Score'], 
                                    bins=[0, 60, 70, 80, 90, 100], 
                                    labels=['0-60', '61-70', '71-80', '81-90', '91-100'])
                success_by_range = mapped_df.groupby(score_ranges)['ICD_Code'].apply(lambda x: x.notna().sum())
                total_by_range = mapped_df.groupby(score_ranges).size()
                success_rate_by_range = (success_by_range / total_by_range * 100).fillna(0)
                
                fig_bar = px.bar(
                    x=success_rate_by_range.index.astype(str),
                    y=success_rate_by_range.values,
                    title="Success Rate by Score Range",
                    labels={'x': 'Score Range', 'y': 'Success Rate (%)'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No match score data available for visualization")
            
            # Top ICD categories
            st.subheader("🏆 Most Common ICD Categories")
            if 'ICD_Code' in mapped_df.columns:
                valid_codes = mapped_df[mapped_df['ICD_Code'].notna() & (mapped_df['ICD_Code'] != '')]
                if len(valid_codes) > 0:
                    icd_categories = valid_codes['ICD_Code'].str[0].value_counts().head(10)
                    if len(icd_categories) > 0:
                        fig_pie = px.pie(
                            values=icd_categories.values,
                            names=icd_categories.index,
                            title="Top 10 ICD-10 Categories"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No valid ICD categories found for analysis")
                else:
                    st.info("No valid ICD codes found for category analysis")
        else:
            st.info("No analytics data available. Upload mapped diagnoses with Match_Score column to see charts.")
    
    with tab5:
        data_preparation_section()

if __name__ == "__main__":
    main()