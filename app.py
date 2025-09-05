import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Multi-Model Evaluation Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple authentication
def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["username"] == "FIRSTSOURCE123" and 
            st.session_state["password"] == "FIRSTSOURCE123"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show login form
        st.markdown("## 🔐 Dashboard Login")
        st.markdown("Please enter your credentials to access the dashboard:")
        
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submit_button = st.form_submit_button("Login", on_click=password_entered)
        
        st.info("Contact your administrator for access credentials.")
        return False
        
    elif not st.session_state["password_correct"]:
        # Password incorrect, show login form again
        st.markdown("## 🔐 Dashboard Login")
        st.error("Username or password incorrect. Please try again.")
        
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submit_button = st.form_submit_button("Login", on_click=password_entered)
        
        st.info("Contact your administrator for access credentials.")
        return False
        
    else:
        # Password correct
        return True

# Check authentication first
if check_password():
    # Add logout button in sidebar
    with st.sidebar:
        st.markdown("### Welcome!")
        if st.button("🚪 Logout"):
            st.session_state["password_correct"] = False
            st.rerun()

    # Custom CSS
    st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        
        .main-header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .data-status {
            background: rgba(40, 167, 69, 0.1);
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid rgba(40, 167, 69, 0.3);
            margin-bottom: 2rem;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 15px;
            background: rgba(255,255,255,0.8);
            border-radius: 20px;
            border: 1px solid #ddd;
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        
        .legend-color {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .mortgage-section {
            background: rgba(255, 243, 205, 0.5);
            padding: 1.5rem;
            border-radius: 15px;
            border: 2px solid rgba(255, 193, 7, 0.3);
            margin: 2rem 0;
        }
        
        /* Hide streamlit menu */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Model configurations - 6 MODELS INCLUDING NEW V2_BASE_CPT_RESIDUAL_DPO_RUN1
    MODEL_COLORS = {
        'MODEL A': '#FF6B6B',
        'MODEL B': '#4ECDC4', 
        'MODEL C': '#45B7D1',
        'MODEL D': '#FECA57',
        'MODEL F': '#8B5CF6',
        'MODEL J': '#32CD32'  # New model - Light Green
    }

    MODEL_NAMES = {
        'MODEL A': 'LLAMA 3.1 8B INSTRUCT',
        'MODEL B': 'V1_INSTRUCT_SFT_CK34',
        'MODEL C': 'V2_BASE_CPT_SFT_CK21',
        'MODEL D': 'V2_BASE_CPT_SFT_DPO_RUN1',
        'MODEL F': 'V2_BASE_CPT_RESIDUAL',
        'MODEL J': 'V2_BASE_CPT_RESIDUAL_DPO_RUN1'  # New model
    }

    # Mortgage Terms Model configurations
    MORTGAGE_COLORS = {
        'LLaMA': '#FF9800',  # Orange
        'DPO': '#9C27B0'     # Purple
    }

    MORTGAGE_NAMES = {
        'LLaMA': 'LLaMA3.1-8B-Instruct',
        'DPO': 'DPO Model'
    }

    # Column mappings - UPDATED FOR NEW 6-MODEL STRUCTURE
    COLUMN_MAPPINGS = {
        'qa': {
            'judgeColumns': [
                'Judge_Model_A_Score_New',  # New judge scores for QA
                'Judge_Model_B_Score_New',
                'Judge_Model_C_Score_New',
                'Judge_Model_H_Score_New',  # This maps to MODEL D
                'Judge_Model_F_Score_New',
                'Judge_Model_J_Score_New'   # New model
            ],
            'bertColumns': [
                'f1_base',                                          # MODEL A
                'f1_V34',                                          # MODEL B  
                'bertscore_f1_v21',                                # MODEL C
                'bertscore_f1_v2_dpo_run1',                        # MODEL D
                'bertscore_f1_v2_cpt_residual',                    # MODEL F
                'bertscore_f1_V2_BASE_CPT_RESIDUAL_DPO_RUN1'       # MODEL J (New)
            ]
        },
        'summary': {
            'judgeColumns': [
                'Judge_Model_A_Score_New',  # New judge scores for summary
                'Judge_Model_B_Score_New',
                'Judge_Model_C_Score_New',
                'Judge_Model_H_Score_New',  # This maps to MODEL D
                'Judge_Model_F_Score_New',
                'Judge_Model_J_Score_New'   # New model
            ],
            'bertColumns': [
                'instruct_bertscore_f1',                           # MODEL A
                'finetune_bertscore_f1',                           # MODEL B
                'sft_v21_bertscore_f1',                            # MODEL C
                'bertscore_f1_v2_dpo_run1',                        # MODEL D
                'bertscore_f1_v2_cpt_residual',                    # MODEL F
                'bertscore_f1_V2_BASE_CPT_RESIDUAL_DPO_RUN1'       # MODEL J (New)
            ]
        },
        'classification': {
            'judgeColumns': [
                'Judge_Model_A_Score_New',  # New judge scores for classification
                'Judge_Model_B_Score_New',
                'Judge_Model_C_Score_New',
                'Judge_Model_H_Score_New',  # This maps to MODEL D
                'Judge_Model_F_Score_New',
                'Judge_Model_J_Score_New'   # New model
            ],
            'bertColumns': [
                'instruct_bertscore_f1',                           # MODEL A
                'finetune_bertscore_f1',                           # MODEL B
                'sft_v21_bertscore_f1',                            # MODEL C
                'bertscore_f1_v2_dpo_run1',                        # MODEL D
                'bertscore_f1_v2_cpt_residual',                    # MODEL F
                'bertscore_f1_V2_BASE_CPT_RESIDUAL_DPO_RUN1'       # MODEL J (New)
            ]
        }
    }

    # Data file paths - CORRECTED to match your actual files
    DATA_FILES = {
        'qa': 'data/qa_data.xlsx',
        'summary': 'data/summary_data.xlsx',
        'classification': 'data/classification_data.xlsx'
    }

    # Mortgage Terms Data Files
    MORTGAGE_FILES = {
        'judge': 'mortgage_terms_judge_evaluation.xlsx',
        'bert': 'mortgage_terms_bertscore_evaluation.xlsx'
    }

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_data_from_server():
        """Load data from server files"""
        datasets = {}
        file_status = {}
        
        for task, file_path in DATA_FILES.items():
            try:
                if os.path.exists(file_path):
                    df = pd.read_excel(file_path)
                    datasets[task] = df
                    file_status[task] = {
                        'status': 'success',
                        'rows': len(df),
                        'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    datasets[task] = None
                    file_status[task] = {'status': 'missing', 'rows': 0, 'last_modified': 'N/A'}
            except Exception as e:
                datasets[task] = None
                file_status[task] = {'status': 'error', 'error': str(e), 'rows': 0, 'last_modified': 'N/A'}
        
        return datasets, file_status

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_mortgage_data():
        """Load mortgage terms evaluation data"""
        mortgage_data = {}
        mortgage_status = {}
        
        for data_type, file_path in MORTGAGE_FILES.items():
            try:
                if os.path.exists(file_path):
                    df = pd.read_excel(file_path)
                    mortgage_data[data_type] = df
                    mortgage_status[data_type] = {
                        'status': 'success',
                        'rows': len(df),
                        'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    mortgage_data[data_type] = None
                    mortgage_status[data_type] = {'status': 'missing', 'rows': 0, 'last_modified': 'N/A'}
            except Exception as e:
                mortgage_data[data_type] = None
                mortgage_status[data_type] = {'status': 'error', 'error': str(e), 'rows': 0, 'last_modified': 'N/A'}
        
        return mortgage_data, mortgage_status

    def calculate_averages(df, columns, score_range=(1, 5), task_name=None):
        """Calculate average scores with proper filtering"""
        averages = []
        for col in columns:
            if not col or col not in df.columns:
                averages.append(0)
                continue
                
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if score_range == (1, 5):
                valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
            else:
                valid_scores = numeric_series[(numeric_series >= 0) & (numeric_series <= 1)].dropna()
            
            averages.append(valid_scores.mean() if len(valid_scores) > 0 else 0)
        
        # Ensure we always have 6 models for consistent display
        while len(averages) < 6:
            averages.append(0)
        
        return averages

    def calculate_mortgage_averages(df, columns, score_range=(1, 5)):
        """Calculate average scores for mortgage models"""
        averages = []
        for col in columns:
            if not col or col not in df.columns:
                averages.append(0)
                continue
                
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if score_range == (1, 5):
                valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
            else:
                valid_scores = numeric_series[(numeric_series >= 0) & (numeric_series <= 1)].dropna()
            
            averages.append(valid_scores.mean() if len(valid_scores) > 0 else 0)
        
        return averages

    def calculate_best_overall_model(datasets):
        """Calculate the best overall model across all tasks"""
        model_scores = {model: [] for model in MODEL_COLORS.keys()}
        
        # Use all available tasks for consistent calculation
        for task, data in datasets.items():
            if data is None or data.empty:
                continue
                
            mapping = COLUMN_MAPPINGS.get(task)
            if not mapping:
                continue

            for index, col in enumerate(mapping['judgeColumns']):
                if not col or index >= len(MODEL_COLORS):
                    continue
                    
                model_key = list(MODEL_COLORS.keys())[index]
                
                if col in data.columns:
                    numeric_series = pd.to_numeric(data[col], errors='coerce')
                    valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
                    
                    if len(valid_scores) > 0:
                        model_scores[model_key].extend(valid_scores.tolist())

        best_model = 'MODEL A'
        best_score = 0

        for model, scores in model_scores.items():
            if len(scores) > 0:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model

        return {'model': best_model, 'score': best_score}

    def create_mortgage_judge_chart(mortgage_data):
        """Create mortgage terms judge scores comparison chart"""
        if mortgage_data is None or 'judge' not in mortgage_data or mortgage_data['judge'] is None:
            return go.Figure()
        
        df = mortgage_data['judge']
        judge_columns = ['Judge_LLaMA_Score', 'Judge_DPO_Score']
        model_labels = [MORTGAGE_NAMES['LLaMA'], MORTGAGE_NAMES['DPO']]
        
        averages = calculate_mortgage_averages(df, judge_columns, (1, 5))
        colors = [MORTGAGE_COLORS['LLaMA'], MORTGAGE_COLORS['DPO']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Mortgage Terms',
            x=model_labels,
            y=averages,
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=2,
            text=[f'{avg:.2f}' for avg in averages],
            textposition='outside',
            textfont=dict(size=12, color='black')
        ))
        
        fig.update_layout(
            title="Mortgage Terms - Judge Scores Comparison (1-5 Scale)",
            xaxis_title="Models",
            yaxis_title="Judge Score (1-5 Scale)",
            yaxis=dict(range=[0, 5], dtick=0.5),
            showlegend=False,
            height=400,
            template="plotly_white",
            font=dict(size=10)
        )
        
        return fig

    def create_mortgage_bert_chart(mortgage_data):
        """Create mortgage terms BERT F1 scores comparison chart"""
        if mortgage_data is None or 'bert' not in mortgage_data or mortgage_data['bert'] is None:
            return go.Figure()
        
        df = mortgage_data['bert']
        bert_columns = ['BERTScore_LLaMA_F1', 'BERTScore_DPO_F1']
        model_labels = [MORTGAGE_NAMES['LLaMA'], MORTGAGE_NAMES['DPO']]
        
        averages = calculate_mortgage_averages(df, bert_columns, (0, 1))
        colors = [MORTGAGE_COLORS['LLaMA'], MORTGAGE_COLORS['DPO']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Mortgage Terms',
            x=model_labels,
            y=averages,
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=2,
            text=[f'{avg:.2f}' for avg in averages],
            textposition='outside',
            textfont=dict(size=12, color='black')
        ))
        
        fig.update_layout(
            title="Mortgage Terms - BERT F1 Scores",
            xaxis_title="Models",
            yaxis_title="BERT F1 Score",
            yaxis=dict(range=[0.5, 0.8], dtick=0.05),
            showlegend=False,
            height=400,
            template="plotly_white",
            font=dict(size=10)
        )
        
        return fig

    def create_judge_comparison_chart(datasets, specific_task=None):
        """Create judge scores comparison chart with full model names"""
        fig = go.Figure()
        
        tasks_to_process = [specific_task] if specific_task else [k for k, v in datasets.items() if v is not None]
        
        max_models = 6  # Always show 6 models (A, B, C, D, F, J)
        model_labels = [MODEL_NAMES[list(MODEL_COLORS.keys())[i]] for i in range(max_models)]
        
        for task in tasks_to_process:
            if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
                continue
            
            data = datasets[task]
            mapping = COLUMN_MAPPINGS[task]
            
            averages = calculate_averages(data, mapping['judgeColumns'], (1, 5), task)
            while len(averages) < max_models:
                averages.append(0)
            
            task_colors = {
                'qa': 'rgba(255, 107, 107, 0.8)',
                'summary': 'rgba(78, 205, 196, 0.8)',
                'classification': 'rgba(69, 183, 209, 0.8)'
            }
            
            border_colors = {
                'qa': '#FF6B6B',
                'summary': '#4ECDC4',
                'classification': '#45B7D1'
            }
            
            fig.add_trace(go.Bar(
                name=task.capitalize(),
                x=model_labels,
                y=averages,
                marker_color=task_colors.get(task, 'rgba(128, 128, 128, 0.8)'),
                marker_line_color=border_colors.get(task, '#808080'),
                marker_line_width=2,
                text=[f'{avg:.2f}' for avg in averages],
                textposition='outside',
                textfont=dict(size=10, color='black')
            ))
        
        fig.update_layout(
            title="Judge Scores Comparison (1-5 Scale)",
            xaxis_title="Models",
            yaxis_title="Judge Score (1-5 Scale)",
            yaxis=dict(range=[0, 5], dtick=0.5),
            showlegend=not specific_task,
            height=400,
            template="plotly_white",
            font=dict(size=9),
            xaxis=dict(tickangle=45)
        )
        
        return fig

    def calculate_bert_averages(df, columns, task_name, score_range=(0, 1)):
        """Calculate BERT averages"""
        averages = []
        
        for i, col in enumerate(columns):
            if not col or col not in df.columns:
                averages.append(0)
                continue
                
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if score_range == (1, 5):
                valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
            else:
                valid_scores = numeric_series[(numeric_series >= 0) & (numeric_series <= 1)].dropna()
            
            avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
            averages.append(avg_score)
        
        # Ensure we always have 6 models for consistent display
        while len(averages) < 6:
            averages.append(0)
        
        return averages

    def create_bert_comparison_chart(datasets, specific_task=None):
        """Create BERT F1 scores comparison chart as bar chart with full model names and exact values"""
        fig = go.Figure()
        
        tasks_to_process = [specific_task] if specific_task else [k for k, v in datasets.items() if v is not None]
        
        max_models = 6  # Always show 6 models (A, B, C, D, F, J)
        model_labels = [MODEL_NAMES[list(MODEL_COLORS.keys())[i]] for i in range(max_models)]
        
        for task in tasks_to_process:
            if datasets[task] is None or datasets[task].empty or task not in COLUMN_MAPPINGS:
                continue
            
            data = datasets[task]
            mapping = COLUMN_MAPPINGS[task]
            
            averages = calculate_bert_averages(data, mapping['bertColumns'], task, (0, 1))
            while len(averages) < max_models:
                averages.append(0)
            
            # Use model-specific colors instead of task colors
            if specific_task:  # Individual task page - use model colors
                colors = [MODEL_COLORS[list(MODEL_COLORS.keys())[i]] for i in range(max_models)]
                
                fig.add_trace(go.Bar(
                    name=task.capitalize(),
                    x=model_labels,
                    y=averages,
                    marker_color=colors,
                    marker_line_color='white',
                    marker_line_width=1,
                    text=[f'{avg:.2f}' for avg in averages],
                    textposition='outside',
                    textfont=dict(size=10, color='black')
                ))
            else:  # Overview page - keep task colors for comparison
                task_colors = {
                    'qa': 'rgba(150, 206, 180, 0.8)',
                    'summary': 'rgba(255, 159, 67, 0.8)', 
                    'classification': 'rgba(153, 102, 255, 0.8)'
                }
                
                border_colors = {
                    'qa': '#96CEB4',
                    'summary': '#FF9F43',
                    'classification': '#9966FF'
                }
                
                fig.add_trace(go.Bar(
                    name=task.capitalize(),
                    x=model_labels,
                    y=averages,
                    marker_color=task_colors.get(task, 'rgba(128, 128, 128, 0.8)'),
                    marker_line_color=border_colors.get(task, '#808080'),
                    marker_line_width=2,
                    text=[f'{avg:.2f}' for avg in averages],
                    textposition='outside',
                    textfont=dict(size=10, color='black')
                ))
        
        fig.update_layout(
            title="BERT F1 Scores",
            xaxis_title="Models",
            yaxis_title="BERT F1 Score",
            yaxis=dict(
                range=[0.5, 0.8],
                dtick=0.05,
                tickmode='linear',
                fixedrange=True,
                autorange=False
            ),
            showlegend=not specific_task,
            height=500,
            template="plotly_white",
            font=dict(size=9),
            xaxis=dict(tickangle=45)
        )
        
        return fig

    def create_task_comparison_chart(datasets):
        """Create task performance comparison chart"""
        fig = go.Figure()
        
        valid_tasks = [task for task, data in datasets.items() if data is not None and not data.empty]
        if not valid_tasks:
            return fig
        
        max_models = 6  # Always show 6 models (A, B, C, D, F, J)
        models = list(MODEL_COLORS.keys())[:max_models]
        
        for index, model in enumerate(models):
            task_scores = []
            
            for task in valid_tasks:
                data = datasets[task]
                if task in COLUMN_MAPPINGS:
                    mapping = COLUMN_MAPPINGS[task]
                    
                    if index < len(mapping['judgeColumns']) and mapping['judgeColumns'][index]:
                        col = mapping['judgeColumns'][index]
                        if col in data.columns:
                            numeric_series = pd.to_numeric(data[col], errors='coerce')
                            valid_scores = numeric_series[(numeric_series >= 1) & (numeric_series <= 5)].dropna()
                            avg_score = valid_scores.mean() if len(valid_scores) > 0 else 0
                        else:
                            avg_score = 0
                    else:
                        avg_score = 0
                else:
                    avg_score = 0
                    
                task_scores.append(avg_score)
            
            fig.add_trace(go.Bar(
                name=MODEL_NAMES[model],
                x=[task.capitalize() for task in valid_tasks],
                y=task_scores,
                marker_color=MODEL_COLORS[model],
                opacity=0.8,
                marker_line_color=MODEL_COLORS[model],
                marker_line_width=2,
                text=[f'{score:.2f}' for score in task_scores],
                textposition='outside',
                textfont=dict(size=10, color='black')
            ))
        
        fig.update_layout(
            title="Task Performance Comparison",
            xaxis_title="Tasks",
            yaxis_title="Average Judge Score (1-5 Scale)",
            yaxis=dict(range=[0, 5]),
            height=500,
            template="plotly_white",
            barmode='group',
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=1, font=dict(size=9)),
            font=dict(size=9)
        )
        
        return fig

    # Main dashboard content
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Multi-Model Evaluation Dashboard</h1>
        <p>Comprehensive Analysis with Judge Scores (1-5 Scale) & BERT F1 Scores</p>
        <p><em>Evaluated for QnA, Summary and Classification Tasks on 6 models + Mortgage Terms on 2 models</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data from server
    datasets, file_status = load_data_from_server()
    mortgage_data, mortgage_status = load_mortgage_data()
    
    # Data status display
    st.markdown('<div class="data-status">', unsafe_allow_html=True)
    st.markdown("### Data Status")
    
    # Main datasets status
    col1, col2, col3 = st.columns(3)
    for i, (task, status) in enumerate(file_status.items()):
        with [col1, col2, col3][i]:
            if status['status'] == 'success':
                st.success(f"✅ {task.upper()}: {status['rows']} rows")
                st.caption(f"Last updated: {status['last_modified']}")
            elif status['status'] == 'missing':
                st.warning(f"⚠️ {task.upper()}: File not found")
            else:
                st.error(f"❌ {task.upper()}: {status.get('error', 'Unknown error')}")
    
    # Mortgage datasets status
    st.markdown("#### Mortgage Terms Data")
    col4, col5 = st.columns(2)
    mortgage_items = list(mortgage_status.items())
    
    for i, (data_type, status) in enumerate(mortgage_items):
        with [col4, col5][i]:
            if status['status'] == 'success':
                st.success(f"✅ MORTGAGE {data_type.upper()}: {status['rows']} rows")
                st.caption(f"Last updated: {status['last_modified']}")
            elif status['status'] == 'missing':
                st.warning(f"⚠️ MORTGAGE {data_type.upper()}: File not found")
            else:
                st.error(f"❌ MORTGAGE {data_type.upper()}: {status.get('error', 'Unknown error')}")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main dashboard (only show if we have data)
    if any(df is not None for df in datasets.values()):
        # Summary Statistics
        st.markdown("### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        tasks_loaded = sum(1 for df in datasets.values() if df is not None)
        total_samples = sum(len(df) for df in datasets.values() if df is not None)
        best_model_info = calculate_best_overall_model(datasets)
        mortgage_samples = sum(len(df) for df in mortgage_data.values() if df is not None)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Samples</h4>
                <div class="metric-value">{total_samples:,}</div>
                <div class="metric-label">Main Tasks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Tasks Loaded</h4>
                <div class="metric-value">{tasks_loaded}</div>
                <div class="metric-label">Out of 3</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Overall Model</h4>
                <div class="metric-value">{best_model_info['model']}</div>
                <div class="metric-label">Based on Judge Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Mortgage Samples</h4>
                <div class="metric-value">{mortgage_samples//2 if mortgage_samples > 0 else 0}</div>
                <div class="metric-label">Mortgage Terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Task Selector
        st.markdown("### Analysis Controls")
        available_tasks = ['overview'] + [task for task, df in datasets.items() if df is not None]
        task_options = {
            'overview': 'Overview',
            'qa': 'QA Analysis', 
            'summary': 'Summary Analysis',
            'classification': 'Classification Analysis'
        }
        
        selected_task = st.selectbox(
            "Select Task for Analysis",
            options=available_tasks,
            format_func=lambda x: task_options[x]
        )
        
        task_filter = None if selected_task == 'overview' else selected_task
        
        # Charts Section
        st.markdown("### Visualization Dashboard")
        
        if task_filter:
            st.info(f"Showing analysis for: **{task_filter.title()}** task")
        else:
            st.info("Showing overview across all loaded tasks")
        
        # For overview, show all charts vertically
        if not task_filter:
            # BERT F1 Scores Chart
            st.plotly_chart(create_bert_comparison_chart(datasets, task_filter), use_container_width=True)
            
            # Judge Scores Chart
            st.plotly_chart(create_judge_comparison_chart(datasets, task_filter), use_container_width=True)
            
            # Task Performance Chart (only on overview)
            st.plotly_chart(create_task_comparison_chart(datasets), use_container_width=True)
            
            # Mortgage Terms Section (only on overview)
            if any(df is not None for df in mortgage_data.values()):
                st.markdown('<div class="mortgage-section">', unsafe_allow_html=True)
                st.markdown("### 🏠 Mortgage Terms Model Evaluation")
                st.markdown("**Additional evaluation on mortgage terminology with specialized models**")
                
                # Create two columns for mortgage charts
                mort_col1, mort_col2 = st.columns(2)
                
                with mort_col1:
                    st.plotly_chart(create_mortgage_judge_chart(mortgage_data), use_container_width=True)
                
                with mort_col2:
                    st.plotly_chart(create_mortgage_bert_chart(mortgage_data), use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # For individual tasks, show bert and judge charts vertically
        else:
            # BERT F1 Scores Chart
            st.plotly_chart(create_bert_comparison_chart(datasets, task_filter), use_container_width=True)
            
            # Judge Scores Chart
            st.plotly_chart(create_judge_comparison_chart(datasets, task_filter), use_container_width=True)
        
        # Model Legend moved to bottom
        st.markdown("### Model Legend")
        
        # Main Models Legend
        st.markdown("#### Main Evaluation Models")
        col1, col2 = st.columns(2)
        models_list = list(MODEL_NAMES.items())
        
        for i, (model, name) in enumerate(models_list):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {MODEL_COLORS[model]};"></div>
                    <span><strong>{model}:</strong> {name}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Mortgage Models Legend (only show on overview)
        if not task_filter and any(df is not None for df in mortgage_data.values()):
            st.markdown("#### Mortgage Terms Models")
            col3, col4 = st.columns(2)
            mortgage_models_list = list(MORTGAGE_NAMES.items())
            
            for i, (model, name) in enumerate(mortgage_models_list):
                with col3 if i % 2 == 0 else col4:
                    st.markdown(f"""
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {MORTGAGE_COLORS[model]};"></div>
                        <span><strong>{model}:</strong> {name}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.warning("No data files found. Please ensure the following files exist:")
        st.info("**Main Tasks:**\n• data/qa_data.xlsx\n• data/summary_data.xlsx\n• data/classification_data.xlsx")
        st.info("**Mortgage Terms:**\n• mortgage_terms_judge_evaluation.xlsx\n• mortgage_terms_bertscore_evaluation.xlsx")
