# Data Augmentation for ARC-AGI

## To Do
- [ ] Build Data Visualizer  
- [ ] Increase Examples  
  - Rotate input–output by **90°, 180°, 270°**  
  - Change colors  
  - Flip both input and output  

### Problem Additions
- Rotate only input  
- Rotate only output  
- Horizontal flip  
- Vertical flip  
- Scaling  

### Explore Other Datasets
- Original ARC dataset — **800 tasks**  
- Michael Hodel's RE-ARC dataset — **400 tasks**  
- Simon Strandgaard's PQA dataset — **7 tasks**  
- Simon Strandgaard's Tama dataset — **50 tasks**  
- Mini-ARC — **149 tasks**  
- nosound's handcrafted ARC tasks — **9 tasks**  
- Andy Penrose's tasks — **5 tasks**  

---

## Data Augmentation Pipeline

**Input**  
- Data split: `[train, evaluation, test]`  
- `save_base_dir`: path to save augmented data  
- `file_name` (optional)  

**Output**  
- JSON file containing augmented tasks saved in `save_base_dir`  

**Steps**  
1. Convert grids into NumPy arrays  
2. Apply augmentations:  
   - Rotations (90°, 180°, 270°)  
   - Color changes  
   - Horizontal and vertical flips  
3. Save the updated dataset as JSON in the output directory  

---

## Running the Script

To generate augmented training data, run:

```bash
python3 -m src.sample_augmentation \
  --base_dir data/arc-agi-2025/raw \
  --challenges_filename arc-agi_training_challenges.json \
  --solutions_filename arc-agi_training_solutions.json \
  --save_base_dir data/arc-agi-2025/sample_augmented
```
This will:
1. Load the original ARC-AGI dataset (challenges + solutions)
2. Apply augmentation techniques (rotation, flipping, color changes)
3. Save the augmented dataset in save_base_dir/ provided.