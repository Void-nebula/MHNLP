import pandas as pd

def main():

    df = pd.read_csv('Datasets/HMS_2023-2024_PUBLIC_instchars.csv', encoding_errors='ignore', low_memory=False)
    
    def classify_disorder(row):
        is_depression = 1 if row['dx_dep'] == 1.0 else 0
        is_anxiety = 1 if row['dx_anx'] == 1.0 else 0
        
        if is_depression or is_anxiety:
            return 1
        else:
            return 0

    df['label'] = df.apply(classify_disorder, axis=1)

    race_map = {
        'race_black': 'African American/Black',
        'race_ainaan': 'American Indian or Alaskan Native',
        'race_asian': 'Asian American/Asian',
        'race_his': 'Hispanic/Latin(X)',
        'race_pi': 'Native Hawaiian or Pacific Islander',
        'race_mides': 'Middle Eastern, Arab, or Arab American',
        'race_white': 'White',
        'race_other': 'Other'
    }

    sex_map = {
        1: 'Female',
        2: 'Male',
        3: 'Intersex'
    }

    sexual_orientation_map = {
        'sexual_h': 'Heterosexual',
        'sexual_l': 'Lesbian',
        'sexual_g': 'Gay',
        'sexual_bi': 'Bisexual',
        'sexual_queer': 'Queer',
        'sexual_quest': 'Questioning',
        'sexual_selfID': 'Self-identified',
        'sexual_asexual': 'Asexual',
        'sexual_pan': 'Pansexual',
        'sexual_prefnoresp': 'Preference not to respond'
    }

    fin_map = {
        1: 'Always stressful',
        2: 'Often stressful',
        3: 'Sometimes stressful',
        4: 'Rarely stressful',
        5: 'Never stressful'
    }

    worry_map = {
        1: 'never',
        2: 'sometimes',
        3: 'often'
    }

    degree_map = {
        'degree_ass': 'Associate\'s',
        'degree_bach': 'Bachelor\'s',
        'degree_ma': 'Master\'s',
        'degree_jd': 'JD',
        'degree_md': 'MD',
        'degree_phd': 'PhD',
        'degree_other': 'other',
        'degree_nd': 'non-degree'
    }

    enrollment_map = {
        1: 'full-time',
        2: 'part-time',
        3: 'other'
    }

    gpa_map = {
        'gr_A': 'mostly A\'s',
        'gr_B': 'mostly B\'s',
        'gr_C': 'mostly C\'s',
        'gr_D': 'mostly D\'s',
        'gr_F': 'mostly F\'s',
        'gr_dk': 'no grade'
    }

    aca_impa_map = {
        1: 'none',
        2: '1-2 days',
        3: '3-5 days',
        4: '6 or more days'
    }

    persist_map = {
        1: 'strongly agress',
        2: 'agree', 
        3: 'somewhat agree',
        4: 'somewhat disagree',
        5: 'disagree',
        6: 'strongly disagree'
    }

    impa_map = {
        1: 'not difficult at al',
        2: 'somewhat difficult',
        3: 'very difficult',
        4: 'extremely difficult'
    }

    df['prompt'] = df.apply(lambda row: (
        # Age
        "Age: " + (f"{round(row['age'])}" if (pd.notna(row['age']) and row.get('age') != '') else "Unknown") + '. ' +

        # Sex
        "Sex: " + (f"{sex_map.get(row['sex_birth'], 'Unknown')}" if (pd.notna(row['sex_birth']) and row.get('sex_birth') != '') else "Unknown") + '. ' +

        # Race
        "Race: " + (", ".join([f"{race_map[col]}" for col in race_map if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "Unknown") + '. ' +

        # Sexual orientation
        "Sexual orientation: " + (", ".join([f"{sexual_orientation_map[col]}" for col in sexual_orientation_map if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "Unknown") + '. ' +

        # Financial situation right now
        "Financial situation right now: " + 
        (f"{fin_map.get(row['fincur'], 'Unknown')}" if (pd.notna(row['fincur']) and row.get('fincur') != '') else "Unknown") + '. ' +

        # Financial situation while growing up
        "Financial situation while growing up: " + 
        (f"{fin_map.get(row['finpast'], 'Unknown')}" if (pd.notna(row['finpast']) and row.get('finpast') != '') else "Unknown") + '. ' +

        # Housing worry
        "Within the last 12 months I " +
        (f"{worry_map.get(row['housing_worry'])}" if (pd.notna(row['housing_worry']) and row.get('housing_worry') != '') else "don't know if I") +
        " worried about not having stable housing. " +

        # Degree
        "I am currently enrolled in a degree of " +
        (", ".join([f"{degree_map[col]}" for col in degree_map if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "Unknown") + " program. " +

        # Enrollment status
        "I am currently enrolled as a " +
        (f"{enrollment_map.get(row['enroll'])}" if (pd.notna(row['enroll']) and row.get('enroll') != '') else "Unknown") + " student. " +

        # Overall GPA
        "My current overall GPA is " +
        (", ".join([f"{gpa_map[col]}" for col in gpa_map if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "Unknown") + ". " +

        # Academic performance 
        "I felt that emotional or mental difficulties have hurt my academic performance for " +
        (f"{aca_impa_map.get(row['aca_impa'])}" if (pd.notna(row['aca_impa']) and row.get('aca_impa') != '') else "Unknown days") + " in the last 12 months. " +

        # Academic confidence
        "I " + (f"{persist_map.get(row['persist'])}" if (pd.notna(row['persist']) and row.get('persist') != '') else " am not sure") + " that I am confident I will be able to finish my degree no matter what challenges I may face. " +

        # Sense of belonging
        "I " +
        (f"{persist_map[row['belong1']]}" if (pd.notna(row['belong1']) and row.get('belong1') != '') else "I dont't know how I agree") + 
        " with that I see myself as a part of the campus community. " +

        # Impact of depression
        "It's " +
        (f"{impa_map.get(row['dep_impa'])}" if (pd.notna(row['dep_impa']) and row.get('dep_impa') != '') else "I don't know how difficult") +
        " that PHQ9 questions (depression) made it for me to do my work, take care of things at home, or get along with other people. " +

        # Impact of anxiety
        "It's " +
        (f"{impa_map.get(row['gad7_impa'])}" if (pd.notna(row['gad7_impa']) and row.get('gad7_impa') != '') else "I don't know how difficult") +
        " that GAD7 questions (anxiety) made it for me to do my work, take care of things at home, or get along with other people. "
    ), axis=1)

    df['prompt'] = df['prompt'].str.replace(r'\s+', ' ', regex=True)

    df = df.drop_duplicates()

    final_df = df[['prompt', 'label']]

    output_file_path = 'HMS_2023_processed.csv'

    final_df.to_csv(output_file_path, index=False, na_rep='NA')

if __name__ == "__main__":
    main()