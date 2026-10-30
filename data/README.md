# Data

The dataset (`group_18.csv`, 1,620 rows x 16 columns) is a **simulated dataset supplied by the course** and is not redistributed in this repository.

To re-run the notebook, place the file at `data/group_18.csv`.

| Column | Type | Description |
|---|---|---|
| age | int | Age in years |
| gender | cat | Female / Male |
| sleep_quality_index, brain_fog_level, physical_pain_score, stress_level | float | Self-reported scores (1-10) |
| depression_phq9_score | float | PHQ-9 depression score |
| fatigue_severity_scale_score | float | Fatigue Severity Scale score |
| pem_present, pem_duration_hours | float | Post-exertional malaise (present or not, duration) |
| hours_of_sleep_per_night | float | Hours of sleep |
| work_status, social_activity_level, exercise_frequency, meditation_or_mindfulness | cat | Lifestyle variables |
| **diagnosis** | cat | **Target:** Depression / ME/CFS / Both |

About 1.3% of feature cells are missing (323 in total); the target has no missing values.
