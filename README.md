# ICD-10 Diagnosis Mapper

A Streamlit web application for mapping medical diagnoses to ICD-10 codes using fuzzy matching algorithms.

## Features

- **Single Query**: Map individual diagnoses to ICD-10 codes
- **Batch Processing**: Process multiple diagnoses at once
- **Data Explorer**: Browse existing mappings and ICD codes
- **Analytics Dashboard**: Visualize match scores and success rates
- **File Upload**: Upload your own data files for processing

## Deployment Instructions

### Option 1: Streamlit Community Cloud (Recommended)

1. **Fork/Clone this repository**
2. **Upload your data files** to the repository:
   - `mapped_diagnoses_with_icd.csv` (your exported mappings)
   - `icd10cm_order_2024.txt` (optional - ICD-10 codes file)

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Deploy!

### Option 2: Local Deployment

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

### Option 3: Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t icd10-mapper .
   docker run -p 8501:8501 icd10-mapper
   ```

## Required Files

- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `mapped_diagnoses_with_icd.csv` - Your exported mappings (upload via app)
- `icd10cm_order_2024.txt` - ICD-10 codes file (optional, upload via app)

## File Upload Instructions

If deploying without data files, users can upload their files directly through the web interface:

1. **Mapped Diagnoses CSV**: Should contain columns like `Original_Diagnosis`, `ICD_Code`, `ICD_Description`, `Match_Score`
2. **ICD-10 Codes File**: Text file with ICD codes and descriptions

## Configuration

- **Match Threshold**: Adjustable confidence threshold (50-100%)
- **Number of Alternatives**: How many alternative matches to show (3-10)

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed
2. **File Format Issues**: Check that CSV files have proper headers
3. **Memory Issues**: For large datasets, consider data preprocessing

### Performance Tips:

- Upload clean, preprocessed data for faster loading
- Use the data preparation tools to optimize file formats
- Consider chunking large batch processing jobs

## Support

For issues or questions, please check the troubleshooting section or contact support.


