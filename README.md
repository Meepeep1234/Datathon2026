For: https://github.com/innovateorange/CuseHacksDatathon2026/tree/main

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

- -------------------------
MORE TECHNICAL IMPROVEMENTS
-------------------------------

# New Improvements — What To Do & Why

---

## Switch to ResNet34 instead of ResNet50

ResNet50 has 25 million parameters — think of each parameter as a question the model can ask about an image. With only ~40 images per class, you don't have enough examples to teach the model to use all 25 million questions meaningfully. So instead of learning real flower patterns, it starts memorizing specific photos and their noise. ResNet34 has fewer parameters, meaning fewer questions, meaning it's forced to learn only the most important patterns. Less capacity + noisy small data = ResNet34 wins almost every time.

---

## Use SCE Loss (Symmetric Cross Entropy)

Normal Cross Entropy loss tells the model "be as confident as possible that this is the right answer." The problem is — some of those answers are wrong because the labels are noisy. So the model learns to be very confident about wrong things, which tanks your F1. SCE adds a second term that says "don't be TOO confident" — it essentially punishes the model for memorizing suspicious labels. Think of it like a teacher who says "if you're too sure about something that seems weird, think again." Non-negotiable when the competition explicitly warns labels are noisy.

---

## Two-Stage Training

Right now we stack everything at once — mixup, random erasing, heavy color jitter, dropout, label smoothing — for the entire training run. The problem is all that noise and blending keeps the model in a permanent "I'm not sure" state. It never gets a chance to sharpen up and make clean, confident decisions. So split training into two stages:

- **Stage 1 (majority of training):** Keep mixup ON, moderate augmentation, small label smoothing. This teaches the model general flower patterns without memorizing.
- **Stage 2 (last 10-30 epochs):** Turn mixup OFF, turn erasing OFF, reduce dropout to 0.1, lower the learning rate. This lets the model take everything it learned and sharpen it into clean, precise decisions.

Think of Stage 1 as studying broadly, Stage 2 as doing focused exam prep right before the test.

---

## Freeze BatchNorm After Warmup

BatchNorm is a layer inside the network that normalizes the numbers flowing through it to keep training stable. It keeps a running average of what "normal" looks like. With small noisy data, that running average gets confused and unstable because every batch looks slightly different. After about 10 warmup epochs, freeze those running stats — lock in what "normal" looks like — while still letting the model adjust everything else. Think of it like locking in the camera settings once you've found the right exposure, instead of letting it keep auto-adjusting.

---

## Class-Balanced Sampler

Macro F1 scores all 102 flower classes equally. If your model completely ignores even one rare class, that class gets an F1 of 0, which drags the whole average down hard. Without a balanced sampler, the model naturally focuses on classes it sees most often and essentially forgets rare ones. A class-balanced sampler fixes this by oversampling rare classes — it shows the model rare flowers more often than they naturally appear so every class gets fair attention. Think of it like a teacher making sure the quiet kids in class get called on just as much as the loud ones.

---

## Small-Loss Filtering

After a few warmup epochs, compute the loss on every single training image individually. The images with the highest loss are the ones the model is most confused by — and in a noisy dataset, those are almost always the mislabeled ones. Drop or downweight the top 20-30% highest loss samples each epoch. This stops the model from wasting time trying to learn from bad examples. Think of it like a student ignoring obviously wrong answers in a study guide — why memorize something that's just wrong?

---

## EMA (Exponential Moving Average)

Instead of just using the model's current weights at the end of training, keep a running average of what the weights looked like over the last many epochs. This averaged version of the model is smoother and less sensitive to the noise of any single training step. Almost always adds a small free accuracy gain with zero extra training cost. Think of it like averaging your last 10 test scores instead of just submitting your most recent one — the average is more representative of how good you actually are.

---

## Ensemble Top Checkpoints + TTA

Save multiple checkpoints during training — not just the single best one. At inference time, run each checkpoint on the test images and average all their predictions together. Different checkpoints will be confident about different things, and averaging them out cancels errors. Also use Test Time Augmentation (TTA) — take each test image, flip it, crop it multiple ways, run each version through the model, and average those predictions too. More perspectives on the same image = more accurate final answer. This is the single most reliable way to gain extra Macro F1 points at zero training cost.

- **requirements.txt is mandatory** — run `pip freeze > requirements.txt`, organizers run this before your code

- **Switch predict.py paths before submitting** — INPUT_CSV and IMAGE_DIR must point to test not val before zipping

