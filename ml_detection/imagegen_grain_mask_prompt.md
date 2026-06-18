# Imagegen Grain Mask Prompt

Use this when asking imagegen/vision to create a mask that GeoVue can convert into Cellpose-style polygons.

Input image role: original chip tray compartment image.

Task:
Create a fragmentation mask for the visible individual rock chips/grains in the input image.

Output style:
- Preserve the original image aspect ratio and geometry.
- Output a pure mask image, not an annotated photo.
- Use a solid black background for everything that is not an individual grain.
- Fill each accepted grain as one solid non-black island.
- Use bright distinct flat colors or solid white for grain islands.
- Keep touching grains as separate islands with thin black separation.
- Do not draw text, labels, arrows, legends, shadows, gradients, textures, or photo detail.
- Do not invent grains outside the original chip material.
- Ignore tray/background, ruler bars, labels, empty space, dust smear, and obvious powder unless specifically asked for powder.

Matrix/powder variant:
If matrix or powder is required, use separate filled colors:
- grains: white
- matrix: yellow
- powder/fines: magenta
- background: black

The GeoVue labeler fragments the mask by connected bright islands. It expects: black background; bright filled grain islands; one connected island per grain; black cracks separating touching grains.
