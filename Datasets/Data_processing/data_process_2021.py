import pandas as pd
import argparse

def main():

    df = pd.read_csv('Datasets/HMS_2021-2022_PUBLIC_instchars.csv', encoding_errors='ignore', low_memory=False)
    
    num_rows, num_columns = df.shape

    print("original data number: ", num_rows)

    depression_columns = ['dx_dep_1', 'dx_dep_2', 'dx_dep_3_new', 'dx_dep_4_text_new', 'dx_dep_4_new', 'dx_dep_5_new']

    anxiety_columns = ['dx_ax_1', 'dx_ax_2', 'dx_ax_3', 'dx_ax_4', 'dx_ax_5', 'dx_ax_6_new', 'dx_ax_6_text_new', 'dx_ax_7_new',]

    include_columns_hs = {
        'bar_hs_1': "No need for services",
        'bar_hs_2': "Financial reasons (too expensive, not covered by insurance)",
        'bar_hs_3': "Not enough time",
        'bar_hs_4': "Not sure where to go",
        'bar_hs_5': "Difficulty finding an available appointment",
        'bar_hs_6': "Prefer to deal with issues on my own or with support from family/friends",
        'bar_hs_7': "Privacy concerns",
        'bar_hs_8': "People providing services don’t understand me",
        'bar_hs_9': "Other (please specify)",
        'bar_hs_9_text': "Additional input provided by the user",
        'bar_hs_10': "No barriers [mutually exclusive]",
        'bar_hs_11': "Fear of being mistreated due to my identity/identities",
    }

    include_columns_ns = {
        'bar_ns_1': "I haven’t had the chance to go but I plan to",
        'bar_ns_2': "No need for services",
        'bar_ns_3': "Financial reasons (too expensive, not covered by insurance)",
        'bar_ns_4': "Not enough time",
        'bar_ns_5': "Not sure where to go",
        'bar_ns_6': "Difficulty finding an available appointment",
        'bar_ns_7': "Prefer to deal with issues on my own or with support from family/friends",
        'bar_ns_8': "Other (please specify)",
        'bar_ns_8_text': "Additional input provided by the user",
        'bar_ns_9': "No barriers [mutually exclusive]",
        'bar_ns_10': "Privacy concerns",
        'bar_ns_11': "People providing services don’t understand me",
        'bar_ns_12': "Fear of being mistreated due to my identity/identities"
    }

    include_columns_sib = {
        'sib_cut': "Cut myself",
        'sib_burn': "Burned myself",
        'sib_punch': "Punched or banged myself",
        'sib_scratch': "Scratched myself",
        'sib_pull': "Pulled my hair",
        'sib_bit': "Bit myself",
        'sib_wound': "Interfered with wound healing",
        'sib_carv': "Carved words or symbols into skin",
        'sib_rub': "Rubbed sharp objects into skin",
        'sib_pobj': "Punched or banged an object to hurt myself",
        'sib_other': "Other (please specify)",
        'sib_other_text': "Additional input provided by the user",
        'sib_none': "I'm not hurt myself."
    }

    include_columns_talk = {
        'talk1_1': "Professional clinician (e.g., psychologist, counselor, or psychiatrist)",
        'talk1_2': "Roommate",
        'talk1_3': "Friend (who is not a roommate)",
        'talk1_4': "Significant other/romantic partner",
        'talk1_5': "Family member",
        'talk1_6': "Religious counselor or other religious contact",
        'talk1_7': "Support group",
        'talk1_8': "Other non-clinical source (please specify)",
        'talk1_8_text': "Additional input provided by the user",
        'talk1_9': "I don't talk to anybody."
    }

    include_columns_inf = {
        'inf_1': "Roommate",
        'inf_2': "Friend (who is not a roommate)",
        'inf_3': "Significant other",
        'inf_4': "Family member",
        'inf_5': "Religious counselor or other religious contact",
        'inf_6': "Support group",
        'inf_7': "Other non-clinical source (please specify)",
        'inf_7_text': "Additional input provided by the user",
        'inf_8': "No one",
        'inf_9': "Faculty member/professor",
        'inf_10': "Staff member"
    }

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

    df['text'] = df.apply(lambda row: (
        "The Conversation between doctor and participant, discussing barriers to mental health services: " +
        "In the past 12 months, which of the following factors have caused you to receive fewer services " +
        "(counseling, therapy, or medications) for your mental or emotional health than you would have otherwise received? " +
        "(Select all that apply): " + 
        (", ".join([f"{include_columns_hs[col]}" for col in include_columns_hs if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "no idea") +
        ". " + '''In the past 12 months, which of the following explain why you have not received medication or therapy 
        for your mental or emotional health? (Select all that apply): ''' +
        (", ".join([f"{include_columns_ns[col]}" for col in include_columns_ns if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "no idea") +
        
        # Special handling for 'bar_ns_8_text'
        (f", Additional input provided by the user: {row.get('bar_ns_8_text', '')}" if pd.notna(row.get('bar_ns_8_text', '')) and row.get('bar_ns_8_text', '') != '' else "") +
        
        ". " + '''Instructions for this item: “This question asks about ways you may have hurt yourself on purpose, without
        intending to kill yourself.” In the past year, have you ever done any of the following intentionally? (Select all that apply): ''' +
        (", ".join([f"{include_columns_sib[col]}" for col in include_columns_sib if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "no idea") +
        
        # Special handling for 'sib_other_text'
        (f", Other (please specify): {row.get('sib_other_text', '')}" if pd.notna(row.get('sib_other_text', '')) and row.get('sib_other_text', '') != '' else "") +
        
        ". " + '''If you were experiencing serious emotional distress, whom would you talk to about this? (Select all that apply): ''' +
        (", ".join([f"{include_columns_talk[col]}" for col in include_columns_talk if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "no idea") +
        
        # Special handling for 'talk1_8_text'
        (f", Other non-clinical source (please specify): {row.get('talk1_8_text', '')}" if pd.notna(row.get('talk1_8_text', '')) and row.get('talk1_8_text', '') != '' else "") +
        
        ". " + '''In the past 12 months, have you received support for your mental or emotional health from any of
        the following sources? (Select all that apply): ''' +
        (", ".join([f"{include_columns_inf[col]}" for col in include_columns_inf if pd.notna(row.get(col, '')) and row.get(col, '') != '']) or "no idea") +
        
        # Special handling for 'inf_7_text'
        (f", Other non-clinical source (please specify): {row.get('inf_7_text', '')}" if pd.notna(row.get('inf_7_text', '')) and row.get('inf_7_text', '') != '' else "") +
        
        ". Based on the respondent's answers, is the participant depressed or anxious?"
    ), axis=1)

    df['idx'] = df.index + 1

    df['text'] = df['text'].str.replace('\n', ' ').str.replace('\r', ' ')

    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

    df = df.drop_duplicates()

    final_df = df[['idx', 'text', 'disorder', 'depression_state', 'anxiety_state']]

    output_file_path = '2021.csv'

    final_df.to_csv(output_file_path, index=False, na_rep='NA')

if __name__ == "__main__":
    main()