"""Inputs. Drafts, labels and the edit-outcome table live in the main_v5 tree until release (read-only); CT-RATE in the DATA lake."""
import os

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = f"{V6}/evidence"       # stage 0: one classifier and one measurement table per split
CONFIGS = f"{V6}/configs"
V5 = "/orange/xujie/liang.renjie/P08_TrustCT/main_v5"
CTRATE = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE"
TS_EFFUSION = f"{CTRATE}/segmentation/ts_seg/ts_pleural_pericard_effusion"
