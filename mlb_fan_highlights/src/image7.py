import requests
import base64
from PIL import Image
from io import BytesIO
from google.cloud import secretmanager

def access_secret_version(project_id, secret_id, version_id="latest"):
    """
    Access the secret version from Google Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def generate_image():
    # Get secrets from Google Secret Manager
    try:
        project_id = "gem-rush-007"  # Replace with your Google Cloud project ID
        account_id = access_secret_version(project_id, "cloudflare-account-id")
        api_token = access_secret_version(project_id, "cloudflare-api-token")
    except Exception as e:
        print(f"Error accessing secrets: {e}")
        return

    # API endpoint - using the stable-diffusion-xl-lightning model
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/bytedance/stable-diffusion-xl-lightning"

    # Request headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "prompt": "Gleyber Torres hit a home run, extending the Yankees lead",
        "negative_prompt": "blurry, low quality, distorted",
        "width": 768,
        "height": 768,
        "num_steps": 20,
        "guidance": 7.5
    }

    print("Generating image...")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        # Save the image
        with open("generated_image.jpg", "wb") as f:
            f.write(response.content)
        print("Image saved as 'generated_image.jpg'")
        
        # Optionally display the image if running in a notebook or environment with display capability
        try:
            image = Image.open(BytesIO(response.content))
            image.show()
            print("Image displayed")
        except Exception as e:
            print(f"Could not display image: {e}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    generate_image()



# @cf/bytedance/stable-diffusion-xl-lightning
#@cf/stabilityai/stable-diffusion-xl-base-1.0
# @cf/lykon/dreamshaper-8-lcm
