from data.make_dataset import make_data_set
import click

from models.train_model import create_trained_model
from visualization.visualize import visualize
from models.predict_model import predict


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    make_data_set(input_filepath, output_filepath)
    visualize(output_filepath)
    create_trained_model(output_filepath)
    print(predict("my cat doest like the milk that I bought from the store yesterday"))


if __name__ == "__main__":
    main()
