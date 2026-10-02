# 3-minute demo script

Open `http://<team host>/app` before you start. Have one high-risk clip already in mind as a backup.

**0:00 – 0:25 · Problem.** "Every city intersection, fleet dashcam and warehouse records
video 24/7. Near misses, where someone almost gets hit, are the best predictor of real
accidents, but nobody watches that footage until after an injury. We built an agent that
watches all of it."

**0:25 – 1:15 · Scan.** Click **Scan all cameras**. While it runs: "It sends five near-miss
queries in parallel across every camera in our VAST archive: SF street cams, a Toronto
dashcam, I-24 highway, a neighborhood cam and a warehouse. Search runs over Cosmos Embed
vectors in VastDB, and each clip was described by NVIDIA Cosmos Reason, with YOLO counting
objects." Point at the status line: moments found, cameras, time.

**1:15 – 2:00 · Judge.** Point at the hotspot strip: "A W&B-hosted LLM scored every moment.
SF cam 1 is the hotspot." Click it. Play the top **HIGH** card: it starts at the flagged
second. Read the verdict and the recommended action. Open "Cosmos description" to show the
evidence the LLM used.

**2:00 – 2:40 · Act.** Click **Generate safety briefing**. "This is what lands in the ops
manager's inbox: hotspots, top incidents, three actions."

**2:40 – 3:00 · Stack + close.** "VAST DataEngine and VastDB for ingest and search, NVIDIA
Cosmos Reason and Embed plus YOLO on CoreWeave GPUs, W&B Inference for judging, built in
Cursor. Next: run it continuously on live feeds and page someone on every HIGH."

## Recording

Record the screen at about 2:30 with QuickTime (File → New Screen Recording) as a backup in
case the live demo or Wi-Fi fails.
