import argparse
parser = argparse.ArgumentParser()
parser.add_argument('echo', help="Enter the path to the image")
args = parser.parse_args()
print(args.echo)
