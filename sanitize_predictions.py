import argparse

def clean_prediction(input_file, output_file):

    with open(input_file, "r") as input:
        with open(output_file, "w") as output:

            for line in input.readlines():
                splits = line.strip().split("\t")

                if len(splits) == 2:
                    output.write(f"{splits[0]}\t{splits[1]}\n")
                else:
                    output.write(f"{splits[0]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    clean_prediction(args.input, args.output)
