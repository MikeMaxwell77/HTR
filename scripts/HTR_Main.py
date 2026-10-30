"""Run the modular Better Segmentation v5 workflow."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forms-dir', help='Directory containing IAM form PNGs')
    parser.add_argument('--forms-csv', help='CSV containing Form ID and Form Data')
    parser.add_argument('--model', help='Saved Keras recognition model')
    parser.add_argument('--output', help='Destination prediction CSV')
    parser.add_argument('--no-plots', action='store_true', help='Disable interactive plots')
    args = parser.parse_args()
    # Preserve the original backend default while allowing an environment override.
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    from segmentation.pipeline import run_pipeline
    options = {'show_plots': not args.no_plots}
    for argument, parameter in [('forms_dir', 'forms_dir'), ('forms_csv', 'forms_csv'),
                                ('model', 'model_path'), ('output', 'output_csv')]:
        value = getattr(args, argument)
        if value is not None:
            options[parameter] = value
    run_pipeline(**options)


if __name__ == '__main__':
    main()
