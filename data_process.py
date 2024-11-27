import pandas as pd
import argparse

def main(args):

    df = pd.read_csv('HMS_2022-2023_PUBLIC_instchars.csv', low_memory=False)

    exclude_columns = ['dx_dep_1', 'dx_dep_2', 'dx_dep_3', 'dx_dep_4', 'dx_dep_4_text', 'dx_dep_5']

    if args.generate_strategy == "keep_original":

        if args.task == "binary_class_classification":

            df['label'] = df[exclude_columns].notna().any(axis=1).astype(int)

            prompt_columns = [col for col in df.columns if col not in exclude_columns + ['label']]

            df['text'] = df.apply(lambda row:
                                f"This person has the following characteristics: " + 
                                ", ".join([f"{col}: {row[col]}" for i, col in enumerate(prompt_columns)]) +
                                ". Is this patient depressed?", axis=1)

            df['idx'] = df.index + 1 

            final_df = df[['idx', 'text', 'label']]

            output_file_path = f'{args.generate_strategy}_{args.task}_depression.csv'
            final_df.to_csv(output_file_path, index=False, na_rep='NA')

        else:

            def get_label(row):
                if pd.notna(row['dx_dep_1']):
                    return 1
                elif pd.notna(row['dx_dep_2']):
                    return 2
                elif pd.notna(row['dx_dep_3']):
                    return 3
                elif pd.notna(row['dx_dep_4']) or pd.notna(row['dx_dep_4_text']):
                    return 4
                elif pd.notna(row['dx_dep_5']):
                    return 5
                else:
                    return 0
            
            df['label'] = df.apply(get_label, axis=1)

            prompt_columns = [col for col in df.columns if col not in exclude_columns + ['label']]

            df['text'] = df.apply(lambda row:
                      f"This person has the following characteristics: " + 
                      ", ".join([f"{col}: {row[col]}" for i, col in enumerate(prompt_columns)]) +
                      ". Is this patient depressed?", axis=1)
            
            df['idx'] = df.index + 1 

            final_df = df[['idx', 'text', 'label']]

            output_file_path = f'{args.generate_strategy}_{args.task}_depression.csv'

            final_df.to_csv(output_file_path, index=False, na_rep='NA')


    elif args.generate_strategy == "translate_all":

        df['label'] = df[exclude_columns].notna().any(axis=1).astype(int)

        # gender identity
        gender_columns = {
            'gender_male': 'Male',
            'gender_female': 'Female',
            'gender_transm': 'Trans male/Trans man',
            'gender_transf': 'Trans female/Trans woman',
            'gender_queer': 'Genderqueer/Gender non-conforming',
            'gender_selfid': 'Self-identify (please specify)',
            'gender_text': 'Self-identify (please specify)',
            'gender_nonbin': 'Gender non-binary',
            'gender_prefnoresp': 'Prefer not to respond'
        }

        #  describe your sexual orientation
        gender_description_columns = {
            "sexual_h": "Heterosexual",
            "sexual_l": "Lesbian",
            "sexual_g" : "Gay",
            "sexual_bi": "Bisexual",
            "sexual_queer": "Queer",
            "sexual_quest": "Questioning",
            "sexual_selfid": "Self-identify(please specify)",
            "sexual_text": "",
            "sexual_asexual": "Asexual",
            "sexual_pan": "Pansexual",
            "sexual_prefnoresp": "Prefer not to respond",
        }

        # What is your race/ethnicity?
        race_identity = {
            "race_black": "African American/Black",
            "race_ainaan": "American Indian or Alaskan Native",
            "race_asian": "Asian American/Asian",
            "race_his": "Hispanic/Latin(x)",
            "race_pi": "Native Hawaiian or Pacific Islander",
            "race_mides": "Middle Eastern, Arab, or Arab American",
            "race_white": "White",
            "race_other": "Other race",
            "race_other_text": ""
        }

        # What is your group of race
        race_group = {
            "black_african" : "African",
            "black_africanam": "African American",
            "black_caribean": "Caribbean/West Indian",
            "black_afrolatin": "Afro-Latina/o/x",
            "black_other": "Other",
            "black_other_text": "",
            "asian_east": "East Asian",
            "asian_southeast": "Southeast Asian",
            "asian_south": "South Asian",
            "asian_filipin": "Filipina/o/x",
            "asian_other": "Other",
            "his_mexican": "Mexican/Mexican American",
            "his_centralam": "Central American",
            "his_southam": "South American",
            "his_carribean": "Caribbean",
            "his_spainport": "Spain/Portugal",
            "his_other": "other"
        }

        # International
        international = {
            "international" : "Yes/No",
        }

        # citizeen
        Citizenship = {
            "st_citizen": "US Citizen",
            "st_permanentres": "Permanent Resident/Green Card Holder",
            "st_visa": "A visa holder (F-1, J-1, H1-B, A, L, G, E, and TN)",
            "st_otherdoc": "Other legally documented status (e.g., adjustment of status to permanent Resident) (please specify)",
            "st_notcomfid": "I dont feel comfortable identifying my citizenship status in the U.S.",
            "st_undoc": "Undocumented",
            "st_citizenship_idk": "Temporary Resident/Green Card Holder",
            "st_refugee": "Refugee",
            "st_DACA": "Deferred Action for Childhood Arrivals (DACA)",
            "st_TPS": "Temporary Protected Status (TPS)"
        }

        # age us
        age_US = {
            "value_pair": {
                1 : "U.S.-born",
                2 : "Less than 12 years",
                3 : "12-17 years",
                4 : "18-35 years",
                5 : "More than 35 years",
            }
        }

        fincur = {
            "value_pair": {
                1:"Always stressful",
                2:"Often stressful",
                3:"Sometimes stressful",
                4:"Rarely stressful",
                5:"Never stressful",
            }
        }

        finpast = {
            "value_pair": {
                
            }
        }

        prompt_columns = [col for col in df.columns if col not in exclude_columns + ['label']]

        df['text'] = df.apply(lambda row:
                            f"This person has the following characteristics: " + 
                            ", ".join([f"{col}: {row[col]}" for i, col in enumerate(prompt_columns)]) +
                            ". Is this patient depressed?", axis=1)

        df['idx'] = df.index + 1 

        final_df = df[['idx', 'text', 'label']]

        output_file_path = f'{args.generate_strategy}_depression.csv'
        final_df.to_csv(output_file_path, index=False, na_rep='NA')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a mental health data")
    parser.add_argument("--generate_strategy", default='keep_original',
                    choices=['keep_original','translate_all'], type=str)
    parser.add_argument("--task", default='binary_class_classification',
                    choices=['binary_class_classification','multi_class_classification'], type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

