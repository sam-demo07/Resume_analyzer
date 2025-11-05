# src/ml/skill_matcher.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Fix import paths - remove the problematic import
try:
    from src.parsing import parse_resume
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.parsing import parse_resume

# ----------------------------
# Paths - FIXED
# ----------------------------
# To:
PROJ_ROOT = Path(__file__).resolve().parents[2]  # This points to project root
MODELS_DIR = PROJ_ROOT / "models"
DATASET_CSV = MODELS_DIR / "skills_dataset.csv"
MODEL_PATH = MODELS_DIR / "trained_model.pkl"
VECT_PATH = MODELS_DIR / "vectorizer.pkl"

# ----------------------------
# Skill Synonyms Mapping (Keep your existing)
# ----------------------------
SKILL_SYNONYMS = {
    "python": ["python", "py", "python3", "python programming"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "data visualization": ["data visualization", "visualization", "matplotlib", "seaborn"],
    "sql": ["sql", "mysql", "postgresql"],
    "statistics": ["statistics", "stats"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "eda": ["eda", "exploratory data analysis"],
    "communication": ["communication", "communication skills"],
    "swift": ["swift", "swift programming"],
    "ios": ["ios", "apple ios"],
    "rest": ["rest", "rest api", "restful", "restful api"],
    "soap": ["soap", "soap api"],
    "mvc": ["mvc", "model view controller"],
    "api": ["api", "apis", "web services"],
    "networking": ["networking", "ccna", "routing", "switching"],
    "cloud": ["cloud", "aws", "azure", "gcp"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision", "cv"],
    "mlops": ["mlops", "machine learning operations"],
    "data analysis": ["data analysis", "data analytics"],
    "data engineering": ["data engineering", "data engineer"],
    "devops": ["devops", "development operations"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment"],
    "linux": ["linux", "unix"],
    "infrastructure as code": ["infrastructure as code", "iac"],
    "monitoring": ["monitoring", "system monitoring"],
    "git": ["git", "version control"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs"],
    "vue": ["vue", "vuejs"],
    "node.js": ["node.js", "nodejs", "node"],
    "html/css": ["html", "css", "html5", "css3"],

    # ... keep the rest of your SKILL_SYNONYMS
}

def normalize_skill_to_base(skill: str) -> str:
    """Normalize skill to base form"""
    if not skill:
        return ""
    
    skill_lower = skill.lower().strip()
    skill_lower = re.sub(r'[^\w\s]', '', skill_lower)
    skill_lower = re.sub(r'\s+', ' ', skill_lower)
    
    for base_skill, synonyms in SKILL_SYNONYMS.items():
        if skill_lower in synonyms or skill_lower == base_skill:
            return base_skill
    
    return skill_lower

def debug_skill_matching(resume_skills: List[str], required_skills: List[str]):
    """Debug function to see skill matching process"""
    print("=== DEBUG SKILL MATCHING ===")
    print(f"Original resume skills: {resume_skills}")
    print(f"Original required skills: {required_skills}")
    
    normalized_resume = [normalize_skill_to_base(s) for s in resume_skills]
    normalized_required = [normalize_skill_to_base(s) for s in required_skills]
    
    print(f"Normalized resume: {normalized_resume}")
    print(f"Normalized required: {normalized_required}")
    
    matched = set(normalized_resume) & set(normalized_required)
    missing = set(normalized_required) - set(normalized_resume)
    
    print(f"Matched: {matched}")
    print(f"Missing: {missing}")
    print("========================")

# ----------------------------
# Data loaders - SIMPLIFIED
# ----------------------------
def load_skill_dataset(csv_path: Path = DATASET_CSV) -> Dict[str, List[str]]:
    """Load CSV with skills dataset - FIXED VERSION"""
    print(f"🔍 Looking for CSV at: {csv_path}")
    print(f"🔍 CSV exists: {csv_path.exists()}")
    
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            roles = {}
            print(f"✅ CSV loaded successfully with {len(df)} rows")
            
            for _, row in df.iterrows():
                role = row['role']
                skills_str = row['skills']
                # Handle both string and list types for skills
                if isinstance(skills_str, str):
                    skills = [s.strip() for s in skills_str.split(';') if s.strip()]
                else:
                    skills = skills_str
                roles[role] = skills
                print(f"   - {role}: {len(skills)} skills")
            
            print(f"✅ Total roles loaded from CSV: {len(roles)}")
            return roles
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            # Fallback to ensure we have data
            return get_fallback_roles()
    
    else:
        print(f"❌ CSV file not found at: {csv_path}")
        return get_fallback_roles()

def get_fallback_roles():
    """Fallback roles when CSV is not available - IMPROVED"""
    print("⚠️ Using fallback skill dataset")
    return {
        'Junior Data Scientist': ['Python', 'Pandas', 'Numpy', 'Data Visualization', 'SQL', 'Statistics', 'Scikit-learn', 'EDA', 'Communication', 'Machine Learning'],
        'Data Analyst': ['SQL', 'Excel', 'Data Visualization', 'Statistics', 'Reporting', 'Communication', 'Tableau', 'Python', 'Data Cleaning'],
        'Software Engineer': ['Python', 'Java', 'Git', 'Algorithms', 'Data Structures', 'OOP', 'Debugging', 'Testing', 'APIs'],
        'DevOps Engineer': ['Docker', 'Linux', 'AWS', 'CI/CD', 'Scripting', 'Kubernetes', 'Monitoring', 'Infrastructure', 'Automation']
    }

def parse_resume_structured(file_path: str) -> Dict[str, List[str]]:
    """Parse resume and return structured data - FIXED VERSION"""
    try:
        result = parse_resume(file_path)
        
        if result and isinstance(result, dict):
            skills = result.get("skills", [])
            # Ensure skills is always a list
            if isinstance(skills, str):
                skills = [skills]
            
            return {
                "skills": skills,
                "education": result.get("education", []),
                "experience": result.get("experience", [])
            }
        
        return {"skills": [], "education": [], "experience": []}
        
    except Exception as e:
        print(f"Error in parse_resume_structured: {e}")
        return {"skills": [], "education": [], "experience": []}

def compute_skill_gap(resume_skills: List[str], required_skills: List[str]) -> Dict[str, List[str]]:
    """Compute skill gap with proper normalization"""
    normalized_resume = set()
    for skill in resume_skills:
        normalized = normalize_skill_to_base(skill)
        if normalized:
            normalized_resume.add(normalized)
    
    skill_mapping = {}
    normalized_required = set()
    
    for skill in required_skills:
        normalized = normalize_skill_to_base(skill)
        if normalized:
            normalized_required.add(normalized)
            skill_mapping[normalized] = skill
    
    matched_normalized = normalized_resume & normalized_required
    missing_normalized = normalized_required - normalized_resume
    
    matched = [skill_mapping[skill] for skill in matched_normalized]
    missing = [skill_mapping[skill] for skill in missing_normalized]
    
    return {"matched": matched, "missing": missing}
# Add these functions to your existing skill_matcher.py

def create_enhanced_resume_text(skills, education, experience):
    """Combine all resume sections for better prediction"""
    text_parts = []
    
    if skills:
        text_parts.append(f"Skills: {', '.join(skills[:10])}")  # Limit to top 10 skills
    if education:
        # Take first 2 education entries and clean them
        edu_clean = [edu.replace('<br>', ' ').replace('•', '').strip() for edu in education[:2]]
        text_parts.append(f"Education: {', '.join(edu_clean)}")
    if experience:
        # Take first 2 experience entries and clean them
        exp_clean = [exp.replace('<br>', ' ').replace('•', '').strip() for exp in experience[:2]]
        text_parts.append(f"Experience: {', '.join(exp_clean)}")
    
    enhanced_text = ". ".join(text_parts)
    print(f"🔍 Enhanced resume text for ML: {enhanced_text[:200]}...")
    return enhanced_text

def get_ml_predictions_enhanced(resume_text, roles_map):
    """Get ML predictions using enhanced features"""
    if not MODEL_PATH.exists() or not VECT_PATH.exists():
        print("❌ ML model files not found")
        return None
    
    try:
        clf = joblib.load(MODEL_PATH)
        vect = joblib.load(VECT_PATH)
        
        # Transform and predict
        X = vect.transform([resume_text])
        probabilities = clf.predict_proba(X)[0]
        predicted_class = clf.predict(X)[0]
        confidence = max(probabilities) * 100
        
        # Get top predictions
        predictions = []
        for i, (class_name, prob) in enumerate(zip(clf.classes_, probabilities)):
            predictions.append((class_name, round(prob * 100, 2)))
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🔍 Enhanced ML Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")
        
        return {
            "predictions": predictions[:5],  # Top 5 predictions
            "predicted_class": predicted_class,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        return None
    
def analyze_resume(file_path: str, chosen_role: Optional[str] = None) -> Dict:
    """End-to-end resume analysis using enhanced ML model"""
    roles_map = load_skill_dataset()
    structured = parse_resume_structured(file_path)

    # DEBUG: Show what was parsed
    print(f"DEBUG: Parsed skills: {len(structured.get('skills', []))} skills")
    print(f"DEBUG: Parsed education: {len(structured.get('education', []))} entries")
    print(f"DEBUG: Parsed experience: {len(structured.get('experience', []))} entries")
    
    skills_list = structured.get("skills", [])
    education_list = structured.get("education", [])
    experience_list = structured.get("experience", [])
    
    # Create enhanced text for ML prediction (using ALL data)
    enhanced_resume_text = create_enhanced_resume_text(skills_list, education_list, experience_list)
    
    # Try enhanced ML prediction first
    ml_result = get_ml_predictions_enhanced(enhanced_resume_text, roles_map)
    
    if ml_result and ml_result["confidence"] > 20:  # Only use ML if somewhat confident
        print("✅ Using enhanced ML predictions")
        predictions = ml_result["predictions"]
        predicted_class = ml_result["predicted_class"]
        ml_confidence = ml_result["confidence"]
    else:
        print("⚠️ ML low confidence, using rule-based fallback")
        # Fallback to rule-based predictions
        predictions = []
        for role, required_skills in roles_map.items():
            gap_result = compute_skill_gap(skills_list, required_skills)
            match_score = (len(gap_result["matched"]) / len(required_skills)) * 100 if required_skills else 0
            predictions.append((role, match_score))
        predictions.sort(key=lambda x: x[1], reverse=True)
        predicted_class = predictions[0][0] if predictions else next(iter(roles_map.keys()))
        ml_confidence = 30.0  # Default low confidence for rule-based
    
    if chosen_role is None:
        chosen_role = predicted_class

    required = roles_map.get(chosen_role, [])
    
    # DEBUG: Show matching process
    print(f"DEBUG: Required skills for {chosen_role}: {len(required)} skills")
    
    gap = compute_skill_gap(skills_list, required)

    total = len(required)
    score = (len(gap["matched"]) / total) * 100 if total > 0 else 0.0

    return {
        "parsed": structured,
        "predictions": predictions[:3],  # Top 3
        "chosen_role": chosen_role,
        "required_skills": required,
        "gap": gap,
        "match_score": score,
        "ml_confidence": ml_confidence,  # Add ML confidence to results
        "using_enhanced_ml": ml_result is not None and ml_result["confidence"] > 20
    }

def get_rule_based_predictions(skills_list: List[str], roles_map: Dict[str, List[str]]) -> List[Tuple[str, float]]:
    """Fallback rule-based predictions when ML model is not available"""
    predictions = []
    for role, required_skills in roles_map.items():
        gap_result = compute_skill_gap(skills_list, required_skills)
        match_score = (len(gap_result["matched"]) / len(required_skills)) * 100 if required_skills else 0
        predictions.append((role, match_score))
    
    # Sort by score
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions