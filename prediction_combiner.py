import re
import pandas as pd


def parse_prediction(log_path, outdir):
    # /Users/ibm/GitRepo/darknet/data/0_8.JPEG: Predicted in 40.548927 seconds.
    start_line_regex = re.compile("^(.*)\.JPEG:\\s+Predicted in ([-+]?[0-9]*\.?[0-9]*) seconds\.\\s+$")
    # ad_female: 25%
    kind_regex = re.compile("^(ad_male|sub_male|juvenile|ad_female): [0-9]*%\\s+$")

    df = pd.DataFrame(columns=['img', 'ad_male', 'sub_male', 'juvenile', 'ad_female'])

    with open(log_path, "r") as log:
        img_path_no_ext = ""
        ad_male_counter, sub_male_counter, juvenile_counter, ad_female_counter = 0, 0, 0, 0
        first_flag = True
        for line in log:
            search_start = start_line_regex.search(line)
            if search_start:
                if first_flag:
                    img_path_no_ext = search_start.group(1)
                    print "parsing" + img_path_no_ext + " This is the first image."
                    first_flag = False
                    continue
                # save count then clear
                # print "Count for different kinds: " + str(ad_male_counter) + " " + str(sub_male_counter) + " " + str(juvenile_counter) + " " + str(ad_female_counter)
                df = df.append({'img': img_path_no_ext, 'ad_male': ad_male_counter, 'sub_male': sub_male_counter, 'juvenile': juvenile_counter,
                                   'ad_female': ad_female_counter}, True)
                img_path_no_ext = search_start.group(1)
                print "parsing" + img_path_no_ext + " Result for the last image is saved to df."
                ad_male_counter, sub_male_counter, juvenile_counter, ad_female_counter = 0, 0, 0, 0
                continue
            kind_line = kind_regex.search(line)
            if kind_line:
                kind = kind_line.group(1)
                # print kind
                if kind == "ad_male":
                    ad_male_counter += 1
                elif kind == "sub_male":
                    sub_male_counter += 1
                elif kind == "juvenile":
                    juvenile_counter += 1
                elif kind == "ad_female":
                    ad_female_counter += 1
        # print "Count for different kinds: " + str(ad_male_counter) + " " + str(sub_male_counter) + " " + str(
        #     juvenile_counter) + " " + str(ad_female_counter)
        df = df.append({'img': img_path_no_ext, 'ad_male': ad_male_counter, 'sub_male': sub_male_counter,
                        'juvenile': juvenile_counter,
                        'ad_female': ad_female_counter}, True)
        return df

def sum_prediction(df):
    img_id_regex = re.compile("^.*/(\\d+_\\d+)$")
    df['img_id'] = df['img'].apply(lambda x: img_id_regex.search(x).group(1))
    return df.groupby("img_id").agg({
        'ad_male' : sum,
        'sub_male': sum,
        'juvenile': sum,
        'ad_female': sum
    })

if __name__ == '__main__':
    result_df = parse_prediction("./prediction_log_example", "./data/prediction")
    sum_all = sum_prediction(result_df)
    print sum_all

    '''
        Need to add a prediction algorithm to predict pup here
    '''

