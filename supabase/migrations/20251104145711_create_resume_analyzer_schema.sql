/*
  # Resume Analyzer Database Schema

  1. New Tables
    - `users`
      - `id` (uuid, primary key, references auth.users)
      - `email` (text, unique)
      - `full_name` (text)
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)
    
    - `resumes`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references users)
      - `file_name` (text)
      - `upload_date` (timestamptz)
      - `parsed_skills` (jsonb)
      - `parsed_education` (jsonb)
      - `parsed_experience` (jsonb)
      - `created_at` (timestamptz)
    
    - `analysis_results`
      - `id` (uuid, primary key)
      - `resume_id` (uuid, references resumes)
      - `predicted_role` (text)
      - `match_score` (numeric)
      - `matched_skills` (jsonb)
      - `missing_skills` (jsonb)
      - `confidence_score` (numeric)
      - `precision_score` (numeric)
      - `recall_score` (numeric)
      - `f1_score` (numeric)
      - `created_at` (timestamptz)
    
    - `model_metrics`
      - `id` (uuid, primary key)
      - `model_version` (text)
      - `overall_accuracy` (numeric)
      - `overall_precision` (numeric)
      - `overall_recall` (numeric)
      - `overall_f1` (numeric)
      - `total_predictions` (int)
      - `recorded_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated users to manage their own data
    - Admin policies for model_metrics
*/

-- Users table
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email text UNIQUE NOT NULL,
  full_name text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own data"
  ON users FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can update own data"
  ON users FOR UPDATE
  TO authenticated
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can insert own data"
  ON users FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = id);

-- Resumes table
CREATE TABLE IF NOT EXISTS resumes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  file_name text NOT NULL,
  upload_date timestamptz DEFAULT now(),
  parsed_skills jsonb DEFAULT '[]'::jsonb,
  parsed_education jsonb DEFAULT '[]'::jsonb,
  parsed_experience jsonb DEFAULT '[]'::jsonb,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE resumes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own resumes"
  ON resumes FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own resumes"
  ON resumes FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own resumes"
  ON resumes FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  resume_id uuid NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
  predicted_role text NOT NULL,
  match_score numeric(5,2) DEFAULT 0,
  matched_skills jsonb DEFAULT '[]'::jsonb,
  missing_skills jsonb DEFAULT '[]'::jsonb,
  confidence_score numeric(5,2) DEFAULT 0,
  precision_score numeric(5,2) DEFAULT 0,
  recall_score numeric(5,2) DEFAULT 0,
  f1_score numeric(5,2) DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own analysis"
  ON analysis_results FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM resumes
      WHERE resumes.id = analysis_results.resume_id
      AND resumes.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert own analysis"
  ON analysis_results FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM resumes
      WHERE resumes.id = analysis_results.resume_id
      AND resumes.user_id = auth.uid()
    )
  );

-- Model metrics table (admin only)
CREATE TABLE IF NOT EXISTS model_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_version text NOT NULL DEFAULT 'v1.0',
  overall_accuracy numeric(5,2) DEFAULT 0,
  overall_precision numeric(5,2) DEFAULT 0,
  overall_recall numeric(5,2) DEFAULT 0,
  overall_f1 numeric(5,2) DEFAULT 0,
  total_predictions int DEFAULT 0,
  recorded_at timestamptz DEFAULT now()
);

ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read metrics"
  ON model_metrics FOR SELECT
  TO authenticated
  USING (true);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_resumes_user_id ON resumes(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_resume_id ON analysis_results(resume_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_recorded ON model_metrics(recorded_at DESC);
