import pandas as pd
import argparse

def main():

    df = pd.read_csv('Datasets/HMS_2017-2018_PUBLIC_instchars.csv', encoding_errors='ignore', low_memory=False)
    
    num_rows, num_columns = df.shape

    print("original data number: ", num_rows)

    depression_columns = ['dx_dep_1', 'dx_dep_2', 'dx_dep_3_new', 'dx_dep_4_text_new', 'dx_dep_4_new', 'dx_dep_5_new']

    anxiety_columns = ['dx_ax_1', 'dx_ax_2', 'dx_ax_3', 'dx_ax_4', 'dx_ax_5', 'dx_ax_6_new', 'dx_ax_6_text_new', 'dx_ax_7_new',]

    def classify_disorder(row):
        is_depressed = row[depression_columns].notna().any()
        is_anxious = row[anxiety_columns].notna().any()
        
        # both : 0, depression: 1, anxiety: 2, none: 3
        if is_depressed and is_anxious:
            return 0
        elif is_depressed:
            return 1
        elif is_anxious:
            return 2
        else:
            return 3

    df['disorder'] = df.apply(classify_disorder, axis=1)

    def classify_depression_state(row):
        # 0: none, 1: Major depressive disorder, 2: Dysthymia, 3: Premenstrual, 4: Other, 5: Don't know
        if pd.notna(row['dx_dep_1']):
            return 1
        elif pd.notna(row['dx_dep_2']):
            return 2
        elif pd.notna(row['dx_dep_3_new']):
            return 3
        elif pd.notna(row['dx_dep_4_new']) or pd.notna(row['dx_dep_4_text_new']):
            return 4
        elif pd.notna(row['dx_dep_5_new']):
            return 5
        else:
            return 0

    df['depression_state'] = df.apply(classify_depression_state, axis=1)

    def classify_anxious_state(row):
        # 0: none, 1: Generalized anxiety disorder, 2: Panic disorder, 3: Agoraphopia, 4: Specific phobia
        # 5: Social anxiety disorder, 6: Other, 7: Don't know
        if pd.notna(row['dx_ax_1']):
            return 1
        elif pd.notna(row['dx_ax_2']):
            return 2
        elif pd.notna(row['dx_ax_3']):
            return 3
        elif pd.notna(row['dx_ax_4']):
            return 4
        elif pd.notna(row['dx_ax_5']):
            return 5
        elif pd.notna(row['dx_ax_6_new']) or pd.notna(row['dx_ax_6_text_new']):
            return 6
        elif pd.notna(row['dx_ax_7_new']):
            return 7
        else:
            return 0

    df['anxiety_state'] = df.apply(classify_anxious_state, axis=1)

    gender_map = {
        1: 'Female',
        2: 'Male',
        3: 'Intersex'
    }

    race = {
        'race_black': 'African American/Black',
        'race_ainaan': 'American Indian or Alaskan Native',
        'race_asian': 'Asian American/Asian',
        'race_his_temp': 'Hispanic/Latin(X)',
        'race_pi': 'Native Hawaiian or Pacific Islander',
        'race_mides': 'Middle Eastern, Arab, or Arab American',
        'race_white': 'White',
        'race_other': 'Other'
    }

    agree_maps = {
        1.0: "Strongly agree",
        2.0: "Agree",
        3.0: "Somewhat agree",
        4.0: "Somewhat disagree",
        5.0: "Disagree",
        6.0: "Strongly disagree"
    }

    frequence_maps = {
        1.0: "Always",
        2.0: "Often",
        3.0: "Sometimes",
        4.0: "Rarely",
        5.0: "Never"
    }

    satisfied_maps = {
        1.0: "very dissatisfied",
        2.0: "dissatisfied",
        3.0: "somewhat dissatisfied",
        4.0: "somewhat satisfied",
        5.0: "satisfied",
        6.0: "very satisfied"
    }

    time_maps = {
        1.0: "Less than 1 hour/week",
        2.0: "1-2 hours/week",
        3.0: "3-5 hours/week",
        4.0: "6-10 hours/week",
        5.0: "11-15 hours/week",
        6.0: "16-20 hours/week",
        7.0: "More than 20 hours/week"
    }

    df['text'] = df.apply(lambda row: (
        '''You are a mental health counselor, currently in a conversation with a college student. 
        This student reported the following informations, symptoms and conditions in a recent mental health survey. ''' +
        # Age
        "Age: " + (f"{round(row['age'])}" if (pd.notna(row['age']) and row.get('age') != '') else "Unknown") + '. ' +

        # Gender
        "Gender: " + (f"{gender_map.get(row['sex_birth'], 'Unknown')}" if (pd.notna(row['sex_birth']) and row.get('sex_birth') != '') else "Unknown") + '. ' +

        # Race
        "Race: " + (", ".join([f"{race[col]}" for col in race if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "Unknown") + '. ' +

        # Financial situation
        "Financial situation: " + 
        (f"{frequence_maps[row['fincur']] + " stressful"}" if (pd.notna(row['fincur']) and row.get('fincur') != '') else "Unknown") + '. ' +

        # Sense of belonging
        (f"{"I " + agree_maps[row['belong1']]}" if (pd.notna(row['belong1']) and row.get('belong1') != '') else "I dont't know how I agree") + 
        " with that I see myself as a part of the campus community. " +

        # time spent on class
        "I usually spend " + 
        (f"{time_maps[row['timeclass']]}" if (pd.notna(row['timeclass']) and row.get('timeclass') != '') else "unknown hours") + " attending class/lab. " +

        # time spent on studying
        "I usually spend " + 
        (f"{time_maps[row['timestud']]}" if (pd.notna(row['timestud']) and row.get('timestud') != '') else "unknown hours") + " studying/doing homework. " +

        # Confidence in academic performance
        (f"{"I " + agree_maps[row['persist']]}" if (pd.notna(row['persist']) and row.get('persist') != '') else "am not sure") +
        " that I am confident I will be able to finish my degree no matter what challenges I may face. " + 

        # Satisfaction with social life
        "I " + 
        (f"{satisfied_maps[row['satisfied_overall']]}" if (pd.notna(row['satisfied_overall']) and row.get('satisfied_overall') != '') else "don't know if I safisfied") + 
        " with my overall social and extracurricular experiences at my school . " +

        # Disabling condition
        (f"{"I have a disabling condition" if (pd.notna(row['disab_2']) and row.get('disab_2') == 'Yes') else "I don't have a disabling condition"}") + ". " +

        # Impact of depression
        (f"{"It's " + row['dep_impa'].lower()}" if (pd.notna(row['dep_impa']) and 
                                            (row.get('dep_impa') == 'Not difficult at all' or
                                             row.get('dep_impa') == 'Somewhat difficult' or
                                             row.get('dep_impa') == 'Very difficult' or
                                             row.get('dep_impa') == 'Extremely difficult'
                                             )) else "I don't know how difficult") +
        " that PHQ9 questions (depression) made it for me to do my work, take care of things at home, or get along with other people. " +

        # Impact of anxiety
        (f"{"It's " + row['gad7_impa'].lower()}" if (pd.notna(row['gad7_impa']) and 
                                            (row.get('gad7_impa') == 'Not difficult at all' or
                                             row.get('gad7_impa') == 'Somewhat difficult' or
                                             row.get('gad7_impa') == 'Very difficult' or
                                             row.get('gad7_impa') == 'Extremely difficult'
                                             )) else "I don't know how difficult") +
        " that GAD7 questions (anxiety) made it for me to do my work, take care of things at home, or get along with other people. " +

        "Based on the candidate's answers, is this candidate having any disorders whether of depression or anxiety?"
    ), axis=1)

    df['text'] = df['text'].str.replace('\n', ' ').str.replace('\r', ' ')

    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

    df = df.drop_duplicates()

    final_df = df[['text', 'disorder', 'depression_state', 'anxiety_state']]

    output_file_path = '2017.csv'

    final_df.to_csv(output_file_path, index=False, na_rep='NA')

if __name__ == "__main__":
    main()