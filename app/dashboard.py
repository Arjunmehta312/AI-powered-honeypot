"""
Streamlit Dashboard for AI-Powered Honeypot Intelligence System.

A comprehensive web interface for analyzing honeypot attack data and threat intelligence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import (
    FEATURES_FILE, ATTACK_CLASSIFIER_MODEL, THREAT_SCORER_MODEL,
    PROCESSED_DATA_FILE
)

# Page configuration
st.set_page_config(
    page_title="Honeypot Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .danger-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed honeypot data."""
    try:
        df = pd.read_csv(PROCESSED_DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("Processed data not found. Please run preprocessing first.")
        return None


@st.cache_data
def load_feature_data():
    """Load engineered features data."""
    try:
        df = pd.read_csv(FEATURES_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("Feature data not found. Please run feature engineering first.")
        return None


@st.cache_resource
def load_models():
    """Load trained ML models."""
    models = {}
    try:
        models['attack_classifier'] = joblib.load(ATTACK_CLASSIFIER_MODEL)
        models['threat_scorer'] = joblib.load(THREAT_SCORER_MODEL)
        feature_file = ATTACK_CLASSIFIER_MODEL.parent / 'feature_columns.pkl'
        models['feature_columns'] = joblib.load(feature_file)
        return models
    except FileNotFoundError:
        st.warning("Models not found. Please train models first.")
        return None


def display_overview(df):
    """Display overview statistics."""
    st.markdown('<p class="main-header">AI-Powered Honeypot Intelligence System</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Attacks", f"{len(df):,}")
    
    with col2:
        st.metric("Unique IPs", f"{df['from'].nunique():,}")
    
    with col3:
        st.metric("Unique Ports", f"{df['port'].nunique():,}")
    
    with col4:
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        st.metric("Date Range", f"{date_range} days")
    
    with col5:
        st.metric("Attack Types", f"{df['protocol_type'].nunique():,}")


def plot_attack_timeline(df):
    """Plot attack timeline."""
    st.subheader("Attack Timeline")
    
    # Group by hour
    timeline = df.groupby(df['timestamp'].dt.floor('H')).size().reset_index()
    timeline.columns = ['timestamp', 'count']
    
    fig = px.line(timeline, x='timestamp', y='count',
                  title='Attacks Over Time',
                  labels={'timestamp': 'Time', 'count': 'Number of Attacks'})
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_attack_distribution(df):
    """Plot attack type distribution."""
    st.subheader("Attack Type Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Protocol distribution
        protocol_counts = df['protocol_type'].value_counts().head(10)
        fig = px.bar(x=protocol_counts.index, y=protocol_counts.values,
                     title='Top 10 Protocol Types',
                     labels={'x': 'Protocol', 'y': 'Count'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attack type pie chart
        if 'attack_type' in df.columns:
            attack_counts = df['attack_type'].value_counts()
            fig = px.pie(values=attack_counts.values, names=attack_counts.index,
                        title='Attack Type Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def plot_temporal_analysis(df):
    """Plot temporal analysis."""
    st.subheader("Temporal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attacks by hour
        hour_counts = df['hour'].value_counts().sort_index()
        fig = px.bar(x=hour_counts.index, y=hour_counts.values,
                     title='Attacks by Hour of Day',
                     labels={'x': 'Hour', 'y': 'Count'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attacks by day of week
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts = df['day_of_week'].value_counts().sort_index()
        fig = px.bar(x=[day_names[i] for i in day_counts.index], y=day_counts.values,
                     title='Attacks by Day of Week',
                     labels={'x': 'Day', 'y': 'Count'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def plot_top_attackers(df):
    """Plot top attacking IPs."""
    st.subheader("Top Attacking IPs")
    
    top_ips = df['from'].value_counts().head(15)
    
    fig = px.bar(x=top_ips.values, y=top_ips.index, orientation='h',
                 title='Top 15 Attacking IP Addresses',
                 labels={'x': 'Number of Attacks', 'y': 'IP Address'})
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_port_analysis(df):
    """Plot port analysis."""
    st.subheader("Port Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top targeted ports
        top_ports = df['port'].value_counts().head(15)
        fig = px.bar(x=top_ports.index, y=top_ports.values,
                     title='Top 15 Targeted Ports',
                     labels={'x': 'Port', 'y': 'Count'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Port category distribution
        if 'port_category' in df.columns:
            port_cat_counts = df['port_category'].value_counts()
            fig = px.pie(values=port_cat_counts.values, names=port_cat_counts.index,
                        title='Port Category Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def plot_severity_analysis(df):
    """Plot severity analysis."""
    if 'severity' not in df.columns:
        return
    
    st.subheader("Threat Severity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution
        severity_counts = df['severity'].value_counts()
        colors = {'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'}
        fig = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            marker=dict(colors=[colors.get(s, '#1f77b4') for s in severity_counts.index])
        )])
        fig.update_layout(title='Threat Severity Distribution', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity by attack type
        if 'attack_type' in df.columns:
            severity_attack = pd.crosstab(df['attack_type'], df['severity'])
            fig = px.bar(severity_attack, barmode='stack',
                        title='Severity by Attack Type')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_threat_intelligence(df):
    """Display threat intelligence section."""
    st.header("Threat Intelligence")
    
    if 'reputation_score' in df.columns:
        # Top malicious IPs
        st.subheader("High-Risk IP Addresses")
        
        ip_threat = df.groupby('from').agg({
            'reputation_score': 'first',
            'attack_type': 'count',
            'severity': lambda x: (x == 'HIGH').sum()
        }).reset_index()
        ip_threat.columns = ['IP Address', 'Reputation Score', 'Total Attacks', 'High Severity Attacks']
        ip_threat = ip_threat.sort_values('Reputation Score', ascending=False).head(20)
        
        # Color code by risk
        def risk_color(score):
            if score >= 70:
                return 'background-color: #f8d7da'
            elif score >= 40:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #d4edda'
        
        styled_df = ip_threat.style.applymap(risk_color, subset=['Reputation Score'])
        st.dataframe(styled_df, use_container_width=True)


def display_model_predictions(df, models):
    """Display model predictions interface."""
    st.header("ML Model Predictions")
    
    if models is None:
        st.warning("Models not loaded. Please train models first.")
        return
    
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Attack Type Classifier")
        if 'attack_type' in df.columns:
            attack_dist = df['attack_type'].value_counts()
            total = len(df)
            for attack_type, count in attack_dist.head(5).items():
                percentage = (count / total) * 100
                st.write(f"**{attack_type}**: {count:,} ({percentage:.1f}%)")
    
    with col2:
        st.info("Threat Severity Scorer")
        if 'severity' in df.columns:
            severity_dist = df['severity'].value_counts()
            total = len(df)
            for severity, count in severity_dist.items():
                percentage = (count / total) * 100
                color_box = {
                    'HIGH': 'danger-box',
                    'MEDIUM': 'warning-box',
                    'LOW': 'success-box'
                }.get(severity, 'metric-card')
                st.markdown(f'<div class="{color_box}"><strong>{severity}</strong>: {count:,} ({percentage:.1f}%)</div>',
                           unsafe_allow_html=True)


def display_analytics(df):
    """Display advanced analytics."""
    st.header("Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payload analysis
        st.subheader("Payload Analysis")
        if 'payload_type' in df.columns:
            payload_dist = df['payload_type'].value_counts()
            fig = px.pie(values=payload_dist.values, names=payload_dist.index,
                        title='Payload Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # IP frequency distribution
        st.subheader("IP Frequency Distribution")
        if 'ip_frequency' in df.columns:
            freq_bins = pd.cut(df['ip_frequency'], bins=[0, 10, 50, 100, 500, 10000],
                              labels=['1-10', '11-50', '51-100', '101-500', '500+'])
            freq_dist = freq_bins.value_counts().sort_index()
            fig = px.bar(x=freq_dist.index, y=freq_dist.values,
                        title='IP Attack Frequency Distribution',
                        labels={'x': 'Attacks per IP', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard function."""
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Overview",
        "Attack Analysis",
        "Temporal Patterns",
        "Threat Intelligence",
        "ML Predictions",
        "Advanced Analytics"
    ])
    
    # Load data
    df = load_feature_data()
    if df is None:
        df = load_data()
    
    if df is None:
        st.error("No data available. Please run the preprocessing pipeline first.")
        return
    
    # Load models
    models = load_models()
    
    # Display selected page
    if page == "Overview":
        display_overview(df)
        plot_attack_timeline(df)
        
        col1, col2 = st.columns(2)
        with col1:
            plot_attack_distribution(df)
        with col2:
            plot_port_analysis(df)
    
    elif page == "Attack Analysis":
        st.header("Attack Analysis")
        plot_attack_distribution(df)
        plot_top_attackers(df)
        plot_port_analysis(df)
    
    elif page == "Temporal Patterns":
        st.header("Temporal Patterns")
        plot_temporal_analysis(df)
        plot_attack_timeline(df)
    
    elif page == "Threat Intelligence":
        display_threat_intelligence(df)
        plot_severity_analysis(df)
    
    elif page == "ML Predictions":
        display_model_predictions(df, models)
        if 'attack_type' in df.columns:
            plot_severity_analysis(df)
    
    elif page == "Advanced Analytics":
        display_analytics(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Dataset Info**\n\n"
        f"Total Records: {len(df):,}\n\n"
        f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}\n\n"
        f"Server Location: {df['country'].iloc[0] if 'country' in df.columns else 'Unknown'}"
    )


if __name__ == "__main__":
    main()
