import matplotlib.pyplot as plt
import pandas as pd


def visualize(processed_file_path):
    text_word_count = []
    summary_word_count = []
    data = pd.read_csv(processed_file_path)
    for i in data['cleaned_text']:
        text_word_count.append(len(i.split()))

    for i in data['cleaned_summary']:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})
    length_df.hist(bins=30)
    plt.show()


if __name__ == "__main__":
    visualize()
