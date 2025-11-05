# src/parsing/ml/train_models.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import joblib
import sys
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.parsing.ml.skill_matcher import load_skill_dataset, MODELS_DIR, MODEL_PATH, VECT_PATH, DATASET_CSV

def create_enhanced_training_samples():
    """Create realistic training samples using ALL CSV data (experience + education)"""
    # Load your CSV with all columns
    df = pd.read_csv(DATASET_CSV)
    samples = []
    
    print(f"📊 Loading enhanced data from CSV with {len(df)} roles")
    
    for _, row in df.iterrows():
        role = row['role']
        skills = [s.strip() for s in row['skills'].split(';')]
        min_experience = row['min_experience_years']
        education = row['education']
        
        # Create multiple realistic variations for each role
        role_samples = create_role_variations(role, skills, min_experience, education)
        samples.extend(role_samples)
        
        print(f"   ✅ Created {len(role_samples)} samples for {role}")
    
    return samples

def create_role_variations(role, skills, min_experience, education):
    """Create multiple realistic resume examples for each role"""
    variations = []
    
    # Different experience levels within the role range
    experience_levels = [
        min_experience,
        min_experience + 1,
        min_experience + 2
    ]
    
    for exp in experience_levels:
        # Create different resume templates using ALL available data
        templates = [
            # Template 1: Professional summary style
            f"{role} with {exp} years of experience. {education}. "
            f"Specialized in {', '.join(skills[:3])}. "
            f"Proficient in {', '.join(skills[3:6])}. "
            f"Strong background in {skills[-1] if skills else 'technical field'}.",
            
            # Template 2: Experience-focused
            f"Experienced {role} with {exp} years in the industry. {education}. "
            f"Key skills include {', '.join(skills[:4])}. "
            f"Expertise in {skills[0]} and {skills[1] if len(skills) > 1 else skills[0]}.",
            
            # Template 3: Project-focused  
            f"{role} background with {exp} years experience. {education}. "
            f"Hands-on experience with {', '.join(skills[2:5])}. "
            f"Developed solutions using {skills[0]}, {skills[1]}. "
            f"Knowledgeable in {', '.join(skills[5:7])}.",
            
            # Template 4: Education-focused
            f"{education} graduate seeking {role} position. "
            f"{exp} years of relevant experience. "
            f"Technical skills: {', '.join(skills[:6])}. "
            f"Additional expertise in {', '.join(skills[6:8])}.",
        ]
        
        for template in templates:
            variations.append({
                "text": template,
                "label": role
            })
    
    return variations

def train_with_enhanced_data():
    """Train using enhanced data with experience and education"""
    # Use enhanced features with experience and education
    samples = create_enhanced_training_samples()
    
    print(f"✅ Training with {len(samples)} enhanced samples")
    print(f"✅ Using experience and education data from CSV")
    
    # Show sample of what we're training on
    print("🔍 Sample training texts:")
    for i, sample in enumerate(samples[:2]):
        print(f"   {i+1}: {sample['text'][:80]}...")
    
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    vect = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)  # Increased features
    X = vect.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, labels)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    
    print(f"✅ Model trained on {len(set(labels))} roles")
    print(f"✅ Using enhanced features: experience + education + skills")
    print(f"✅ Saved model -> {MODEL_PATH}")
    print(f"✅ Saved vectorizer -> {VECT_PATH}")
    
    return clf, vect

if __name__ == "__main__":
    train_with_enhanced_data()