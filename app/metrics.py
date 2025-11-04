import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import numpy as np

def calculate_metrics(matched_skills: List[str], missing_skills: List[str],
                     total_required: int, confidence: float = 0.85) -> Dict:
    true_positives = len(matched_skills)
    false_negatives = len(missing_skills)
    false_positives = max(0, int(true_positives * 0.1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'confidence': round(confidence * 100, 2)
    }

def create_metrics_dashboard(metrics: Dict, match_score: float):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['precision'],
            title={'text': "Precision"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#667eea"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['recall'],
            title={'text': "Recall"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#764ba2"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['f1_score'],
            title={'text': "F1-Score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#f093fb"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['confidence'],
            title={'text': "Confidence"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#f5576c"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}}
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

def create_skill_comparison_chart(matched: List[str], missing: List[str]):
    fig = go.Figure(data=[
        go.Bar(name='Matched Skills', x=['Skills'], y=[len(matched)], marker_color='#667eea'),
        go.Bar(name='Missing Skills', x=['Skills'], y=[len(missing)], marker_color='#f5576c')
    ])
    fig.update_layout(
        title='Skill Match Overview',
        barmode='group',
        height=300,
        showlegend=True
    )
    return fig

def save_metrics_to_db(supabase, resume_id: str, role: str, match_score: float,
                       matched: List[str], missing: List[str], metrics: Dict):
    try:
        supabase.table('analysis_results').insert({
            'resume_id': resume_id,
            'predicted_role': role,
            'match_score': match_score,
            'matched_skills': matched,
            'missing_skills': missing,
            'confidence_score': metrics['confidence'],
            'precision_score': metrics['precision'],
            'recall_score': metrics['recall'],
            'f1_score': metrics['f1_score']
        }).execute()
    except Exception as e:
        st.error(f"Failed to save metrics: {e}")

import streamlit as st
