import fireworks.client
from fireworks.client.image import ImageInference, Answer

# Initialize the ImageInference client
fireworks.client.api_key = "jGQ0lUjQHRq1jfAo2zZRE5fTUsrUx2jfyTTppjknRJ6BwDVy"
inference_client = ImageInference(model="stable-diffusion-xl-1024-v1-0")

def imageGeneration(prompt):
    answer : Answer = inference_client.text_to_image(
        prompt=prompt,
        cfg_scale=10,
        height=1024,
        width=1024,
        sampler=None,
        steps=25,
        seed=3,
        safety_check=False,
        output_image_format="PNG",
        # Add additional parameters here
    )

    if answer.image is None:
        raise RuntimeError(f"No return image, {answer.finish_reason}")
    else:
        answer.image.save("output.png")
    return answer

if __name__ == "__main__":
    prompt = input("Prompt: ")
# Generate an image using the text_to_image method
    image = imageGeneration(prompt)
    # print ("image..............",image)
    # with open('generated_image.png', 'wb') as file:
    #     file.write(image.content)
