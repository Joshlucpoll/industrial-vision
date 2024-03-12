"""
@author: u2139451
"""

import os
import threading
import cv2
import matplotlib.pyplot as plt

# Define the range of parameter values to test
SIGMA_SPACE_VALUES = list(map(lambda x: x / 10, range(0, 50, 1)))
SIGMA_COLOR_VALUES = list(range(0, 500, 10))

DATASET_PATH = "./CBSD68"


def loadDataset():
    # Load the original images
    print("\nLoading original images from dataset...")
    original_images = [
        cv2.imread(f"{DATASET_PATH}/original_png/{i:04d}.png") for i in range(68)
    ]
    print("Original images loaded successfully.\n")

    # Load the noisy images
    print("Loading noisy images from dataset...")
    # Define the noise levels
    noise_levels = [5, 10, 15, 25, 35, 50]
    # Create a dictionary to store the images
    noisy_images = {}
    # Load the noisy images for each noise level
    for level in noise_levels:
        noisy_images[level] = [
            cv2.imread(f"{DATASET_PATH}/noisy{level}/{i:04d}.png") for i in range(68)
        ]

    print("Noisy images loaded successfully.\n")
    print("****************************************************\n")

    return original_images, noisy_images


# Define a function to process the images in parallel
def process_images(
    sigma_color, sigma_space, level, images, original_images, similarity_scores
):
    level_similarities = []
    for i, img in enumerate(images):
        # Apply bilateral filter to image
        filtered_image = cv2.bilateralFilter(img, 15, sigma_color, sigma_space)

        # Compare filtered image to original image
        difference = cv2.absdiff(filtered_image, original_images[i])
        similarity = 1 - (
            difference.sum()
            / (
                filtered_image.shape[0]
                * filtered_image.shape[1]
                * filtered_image.shape[2]
            )
        )

        level_similarities.append(similarity)

    # Store similarity score in the dictionary
    if (sigma_color, sigma_space) not in similarity_scores:
        similarity_scores[(sigma_color, sigma_space)] = {}

    similarity_scores[(sigma_color, sigma_space)][level] = sum(
        level_similarities
    ) / len(level_similarities)


def testBilateralFilter(
    noisy_images, original_images, sigma_color_values, sigma_space_values
):
    # Create variables to keep track of similarity scores
    similarity_scores = {}
    # Create a list to hold the threads
    threads = []

    # Iterate over the parameter values
    total_iterations = len(sigma_color_values) * len(sigma_space_values) + 1
    current_iteration = 0

    for sigma_color in sigma_color_values:
        for sigma_space in sigma_space_values:
            current_iteration += 1
            percentage = (current_iteration / total_iterations) * 100

            print(
                f"Testing Bilateral Filter with sigma color: {sigma_color}, sigma space: {sigma_space} - {percentage:.2f}% complete"
            )

            # Access the images using the dictionary
            for level, images in noisy_images.items():
                # Create a thread for each level of noise
                thread = threading.Thread(
                    target=process_images,
                    args=(
                        sigma_color,
                        sigma_space,
                        level,
                        images,
                        original_images,
                        similarity_scores,
                    ),
                )
                thread.start()
                threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Testing Finished - 100% complete")
    print("\n****************************************************\n")
    return similarity_scores


def aggregateAndPrintSimilarityScores(similarity_scores):
    average_similarity_scores = []

    # Print all results the similarity scores
    print("\nSimilarity scores:")
    for key, value in similarity_scores.items():
        print(f"Sigma color: {key[0]}, Sigma space: {key[1]}")

        print(f"Average similarity score: {sum(value.values()) / len(value)}")
        for level, score in value.items():
            print(f"Noise level {level}: {score}")
        print("\n")

        average_similarity_scores.append(
            [key[0], key[1], sum(value.values()) / len(value)]
        )

    print("****************************************************\n")
    return average_similarity_scores


def printSummary(average_similarity_scores):

    # Print the summary of the results
    print("\nSummary:")
    print(
        "{:<20} {:<20} {:<20}".format(
            "Sigma Colour", "Sigma Space", "Average Similarity (higher is better)"
        )
    )
    for sigma_color, sigma_space, average_similarity in sorted(
        average_similarity_scores, key=lambda x: x[2], reverse=True
    ):
        print(
            "{:<20} {:<20} {:<20}".format(sigma_color, sigma_space, average_similarity)
        )

    best_result = sorted(average_similarity_scores, key=lambda x: x[2], reverse=True)[0]

    # print the best result
    print("\nBest result:")
    print(
        f"Sigma Colour: {best_result[0]} Sigma Space: {best_result[1]} Average Similarity: {best_result[2]}"
    )


def saveData(average_similarity_scores, sigma_color_values, sigma_space_values):
    # Ask the user if they want to save the data
    save_data = input("Do you want to save the data? (y/n): ")

    if save_data.lower() == "y":
        # Create the filename based on the parameter ranges
        filename = f"1_data_output/average_similarity_scores_colour_{min(sigma_color_values)}-{max(sigma_color_values)}_space_{min(sigma_space_values)}-{max(sigma_space_values)}.csv"

        # Save the data to a file
        with open(filename, "w") as file:
            file.write(f"Sigma Colour,Sigma Space,Average Similarity\n")
            for row in average_similarity_scores:
                file.write(f"{row[0]},{row[1]},{row[2]}\n")

        print("Data saved successfully.")
    else:
        print("Data not saved.")


def retrieveData():
    # Get the list of files in the directory
    files = os.listdir("1_data_output")

    # Filter the files to only include those starting with "average_similarity_scores"
    filtered_files = [
        file for file in files if file.startswith("average_similarity_scores")
    ]

    # Print the filtered files
    print("Available files:")
    for i, file in enumerate(filtered_files):
        print(f"{i+1}. {file}")

    # Ask the user to select a file
    selection = input("\nSelect a file (enter the corresponding number): ")

    # Get the selected file
    selected_file = filtered_files[int(selection) - 1]

    # Read the contents of the file
    average_similarity_scores = []
    try:
        with open(f"1_data_output/{selected_file}", "r") as file:
            # Skip the header line
            next(file)
            # Read the remaining lines
            for line in file:
                sigma_color, sigma_space, average_similarity = line.strip().split(",")
                average_similarity_scores.append(
                    [float(sigma_color), float(sigma_space), float(average_similarity)]
                )

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")

    return average_similarity_scores


def showVisualisation(average_similarity_scores):
    # Extract the data for plotting
    sigma_color_values = [row[0] for row in average_similarity_scores]
    sigma_space_values = [row[1] for row in average_similarity_scores]
    average_similarity_values = [row[2] for row in average_similarity_scores]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the data
    ax.scatter(sigma_color_values, sigma_space_values, average_similarity_values)

    # Set labels and title
    ax.set_xlabel("Sigma Color")
    ax.set_ylabel("Sigma Space")
    ax.set_zlabel("Average Similarity")
    ax.set_title("Average Similarity Scores")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Ask the user if they want to retrieve data
    retrieve_data = input("\nDo you want to retrieve data? (y/n): ")

    if retrieve_data.lower() == "y":
        # Retrieve the data
        average_similarity_scores = retrieveData()
        # Print the summary
        printSummary(average_similarity_scores)
        # show the visualisation
        showVisualisation(average_similarity_scores)
    else:
        print("\n")

        # get the original and noisy images from CBSD68 dataset
        original_images, noisy_images = loadDataset()

        # Test the bilateral filter on the images
        similarity_scores = testBilateralFilter(
            noisy_images, original_images, SIGMA_COLOR_VALUES, SIGMA_SPACE_VALUES
        )

        # Aggregate and print the results
        average_similarity_scores = aggregateAndPrintSimilarityScores(similarity_scores)

        # print summary of the results
        printSummary(average_similarity_scores)

        # Save the data
        saveData(average_similarity_scores, SIGMA_COLOR_VALUES, SIGMA_SPACE_VALUES)

        # Show the visualisation
        showVisualisation(average_similarity_scores)
