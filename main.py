import discord
from discord.ext import commands
from discord import app_commands
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
import os
# Replace 'your-discord-bot-token' with your actual bot token
TOKEN = os.environ['TOKEN']

# Create a new bot instance
bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())

@bot.event
async def on_ready():
    await bot.change_presence(status=discord.Status.idle, activity=discord.Game("/auric"))
    print("Bot is Ready")

    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(e)

# Define the image generation function
def generate_image(prompt, model_name, height, width, steps, guidance, negative_prompt):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.safety_checker = None
    image = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance, negative_prompt=negative_prompt).images[0]
    return image

# Define the slash command
@bot.tree.command(name="generate", description="Generate an image using a specified model and prompt")
@app_commands.describe(prompt="The image prompt", model_name="The name of the model to use")
async def generate(interaction: discord.Interaction, prompt: str, model_name: str):
    await interaction.response.defer()

    # Set parameters for image generation
    height = 800
    width = 640
    steps = 25
    guidance = 5
    negative_prompt = "easynegative, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,"

    try:
        # Generate the image
        image = generate_image(prompt, model_name, height, width, steps, guidance, negative_prompt)
        
        # Save image to a BytesIO object
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Send the image back in the response
        file = discord.File(fp=img_buffer, filename='image.png')
        await interaction.followup.send(file=file)
    except Exception as e:
        await interaction.followup.send(content=f"An error occurred: {str(e)}")

# Run the bot
bot.run(TOKEN)

