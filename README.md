Improvements - Practice presenting beforehand, 
Look over color contrast, 
pay attention to audience, 
chunk information on multiple slides. 
Talk slow, waiting, take your time. 
Show humility if you don't know that answer to a question (Thats okay!), 
Being able to translate technical to non technical.

--------------------
TECHNICAL IMPROVEMENTS
---------------------

- **Stay Category 2 always** — your 26% from scratch beat their 90% pretrained on points. Check scoring structure every year but Category 2 is almost certainly always worth more

- **Use pretrained weights only if Category 2 stops being worth more points** — check scoring structure every year before deciding

- **Optimize for Macro F1 from day one** — competition scores F1 not accuracy, save best model based on F1 not accuracy

- **Val accuracy and Macro F1 are very different** — at epoch 114 val acc was 40% but F1 was only 12%. They diverge massively because F1 punishes ignoring rare classes

- **Compute normalization from your training data not ImageNet** — using mean=[0.485, 0.456, 0.406] is technically external knowledge and could violate Category 2

- **Noisy training labels** — competition warned labels contain noise, label smoothing handles this

- **GPU farm only worth it if dataset is much larger** — with 40 images per class your 4070 is already overkill, only consider renting if thousands of images per class

- **Get training running the second you have data** — we spent hours on setup before a single epoch ran, even a broken version tells you what's wrong faster

- **Match folder names to competition EXACTLY before writing code** — val_images vs val/images cost more time than anything else this session

- **Use absolute paths with BASE from day one** — relative paths break depending on where VS Code runs from, set BASE at top of file immediately

- **Use os.path.join for every path** — never use forward slashes in strings on Windows

- **mkdir before running** — create the model/ folder before training or it crashes on the save line

- **Windows breaks num_workers** — set to 0 immediately on Windows, don't debug the multiprocessing error

- **VS Code doesn't always save changes** — always Ctrl+S and verify the line actually changed before running again

- **Install packages to the right Python** — use full path `& "C:/Users/.../python.exe" -m pip install package`

- **Check what's actually in your folders before debugging code** — we thought it was a code problem when the images just weren't there

- **Your GPU is 4070 Laptop with 8GB VRAM not 12GB** — check with gpu_stats() first before assuming specs

- **Epochs burn fast on 4070** — 62 epochs in 16 minutes, set epochs to 1000+ and patience high so it runs the full session

- **SGD not Adam for from scratch training** — SGD with momentum converges better when learning from zero

- **Learning rate 0.1 not 0.00001** — original lr was killing accuracy, way too small for scratch training

- **The original code had broken imports** — mpmath and sympy imported instead of torch and torchvision, always check imports first

- **model_best.pth vs model10.pth** — name model files consistently, predict.py was looking for wrong filename

- **Always give predict.py the full boilerplate** — accidentally submitted 15 lines missing all required code underneath

- **report.pdf is mandatory** — max 2 pages, must state Category 1 or 2, needs preprocessing, architecture, hyperparameters, and citation of outside help

- **requirements.txt is mandatory** — run `pip freeze > requirements.txt`, organizers run this before your code

- **Switch predict.py paths before submitting** — INPUT_CSV and IMAGE_DIR must point to test not val before zipping

