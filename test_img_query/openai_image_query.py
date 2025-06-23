import os
import base64
import argparse
from pathlib import Path
from typing import List, Optional
from openai import OpenAI


class OpenAIImageQueryClient:
    """Client for sending text and multiple images to OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the OpenAI client
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model: OpenAI model to use (default: gpt-4o for vision capabilities)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode an image file to base64 string
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            raise Exception(f"Error encoding image {image_path}: {str(e)}")
    
    def get_image_mime_type(self, image_path: str) -> str:
        """
        Get MIME type for image based on file extension
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MIME type string
        """
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')
    
    def load_images_from_folder(self, folder_path: str, image_names: List[str] = None) -> List[dict]:
        """
        Load images from a preset folder
        
        Args:
            folder_path: Path to the folder containing images
            image_names: List of specific image filenames to load (if None, load all supported images)
            
        Returns:
            List of image dictionaries with base64 data and metadata
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise Exception(f"Folder {folder_path} does not exist")
        
        supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        images = []
        
        if image_names:
            # Load specific images
            for image_name in image_names:
                image_path = folder / image_name
                if not image_path.exists():
                    print(f"Warning: Image {image_name} not found in {folder_path}")
                    continue
                
                if image_path.suffix.lower() in supported_extensions:
                    try:
                        base64_image = self.encode_image_to_base64(str(image_path))
                        mime_type = self.get_image_mime_type(str(image_path))
                        
                        images.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        })
                        print(f"Loaded image: {image_name}")
                    except Exception as e:
                        print(f"Error loading image {image_name}: {str(e)}")
        else:
            # Load all supported images from folder
            for image_path in folder.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in supported_extensions:
                    try:
                        base64_image = self.encode_image_to_base64(str(image_path))
                        mime_type = self.get_image_mime_type(str(image_path))
                        
                        images.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        })
                        print(f"Loaded image: {image_path.name}")
                    except Exception as e:
                        print(f"Error loading image {image_path.name}: {str(e)}")
        
        return images
    
    def send_query(self, text_query: str, images: List[dict], max_tokens: int = 300) -> str:
        """
        Send a query with text and images to OpenAI
        
        Args:
            text_query: Text prompt/question
            images: List of image dictionaries
            max_tokens: Maximum tokens for response
            
        Returns:
            Response from OpenAI
        """
        try:
            # Prepare the content for the message
            content = [{"type": "text", "text": text_query}]
            content.extend(images)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                # max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error sending query to OpenAI: {str(e)}")
    
    def query_with_images_from_folder(self, 
                                    text_query: str, 
                                    folder_path: str, 
                                    image_names: List[str] = None,
                                    max_tokens: int = 300) -> str:
        """
        Convenience method to query with images from a folder
        
        Args:
            text_query: Text prompt/question
            folder_path: Path to folder containing images
            image_names: Specific image filenames to include (if None, includes all)
            max_tokens: Maximum tokens for response
            
        Returns:
            Response from OpenAI
        """
        print(f"Loading images from folder: {folder_path}")
        images = self.load_images_from_folder(folder_path, image_names)
        
        if not images:
            raise Exception("No images were successfully loaded")
        
        print(f"Sending query with {len(images)} images...")
        return self.send_query(text_query, images, max_tokens)


def main():
    """Example usage of the OpenAI Image Query Client"""
    parser = argparse.ArgumentParser(description='Send text and images to OpenAI API')
    parser.add_argument('--folder', required=True, help='Path to folder containing images')
    parser.add_argument('--query', required=True, help='Text query to send')
    parser.add_argument('--images', nargs='*', help='Specific image filenames (optional)')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use')
    parser.add_argument('--max-tokens', type=int, default=300, help='Maximum tokens for response')
    
    args = parser.parse_args()
    
    # Initialize client
    try:
        client = OpenAIImageQueryClient(model=args.model)
        
        # Send query
        response = client.query_with_images_from_folder(
            text_query=args.query,
            folder_path=args.folder,
            image_names=args.images,
            max_tokens=args.max_tokens
        )
        
        print("\n" + "="*50)
        print("OpenAI Response:")
        print("="*50)
        print(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")


# Example usage in code
if __name__ == "__main__":
    # Command line interface
    main()
    
    # Example of direct usage:
    """
    # Create client
    client = OpenAIImageQueryClient()
    
    # Query with all images in folder
    response = client.query_with_images_from_folder(
        text_query="What do you see in these images? Describe each one briefly.",
        folder_path="./images"
    )
    print(response)
    
    # Query with specific images
    response = client.query_with_images_from_folder(
        text_query="Compare these two images and tell me the differences.",
        folder_path="./images",
        image_names=["image1.jpg", "image2.png"]
    )
    print(response)
    
    # Advanced usage - load images separately and send custom query
    images = client.load_images_from_folder("./images", ["photo1.jpg", "photo2.jpg"])
    response = client.send_query(
        text_query="Analyze the content of these images and provide insights.",
        images=images,
        max_tokens=500
    )
    print(response)
    """