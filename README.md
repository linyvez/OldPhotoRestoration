# Old Photo Restoration

## Authors:
- Viktoriia Lushpak (https://github.com/linyvez)
- Viktoriia Koval (https://github.com/Vika-Koval)
- Viktor Pakholok (https://github.com/viktorpakholok)
- Maksym-Vasyl Tarnavskyi (https://github.com/MaVasTarn)

### About
This tool was created to help with fixing and restoring old, faded photos. It uses a simple website interface made with Streamlit, where you can change settings and see the results immediately. The main pipeline includes steps to fix shape of the photo, clean up scratches, correct contrast and add color. You can use full restoration which includes all those steps, or use them separately.

#### Preprocessing
If for example a physical photo was taken at an angle with a phone, the tool uses warping to make the image flat and straight again. It also turns the photo into grayscale which is important for the next steps.

#### Damage Correction
Old photos often have physical problems like scratches, dust, or fold lines. This part of the tool finds those broken lines and cleans them. By fixing the surface first, the final color looks much smoother and more natural. You can calibrate the threshold to get appropriate results.

#### Contrast Correction
This step makes the photo look clearer and sharper using a method called CLAHE. It checks if the photo already has good contrast and uses SSIM score to make sure the fix did not introduce any artifacts. If the score is too low, contrast correction will not be applied.

#### Colorization
Here we can add color back to B&W photos. The main pipeline uses manual colorization. You upload B&W image and pick the colors you want. You draw scribbles on the photo with a brush which will be processed and spread across the whole image. The more precise the scribbles are, the better results you can get. Another method is selective color recovery, which you can use to test the tool using a real color photo. You upload a color image which is hidden underneath its B&W version. When you draw scribbles on the image, it reveals the true colors. The tool then uses these small pieces of original color to try and colorize the rest of the image. Additionally, you can evaluate the results by uploading a color image.