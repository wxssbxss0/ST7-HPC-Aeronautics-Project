# ST7 HPC Aeronautics — Cloud Compute Progress Update
*Written by Wassim — April 2, 2026 ~1AM*

---

## What we were trying to do

Our CentraleSupélec-Metz cluster reservation timed out before we could run the
F-35 full-body Euler and Stokes simulations. Our next reservation isn't until
Friday morning (day of soutenance), which is too late. So tonight I tried to
find an alternative cloud HPC platform to run the sims before then.

---

## What I tried and what happened

### AWS (failed — quota limit)
- Created a fresh AWS account ($100-200 free credits for new accounts)
- Tried to launch a `c6i.32xlarge` (128 vCPUs) — got blocked:
  > "vCPU limit of 16 exceeded" — new accounts are capped at 16 vCPUs by default
- A quota increase request takes 24-48h, too slow for us
- **Abandoned AWS, switched to Google Cloud**

### Google Cloud (working ✅)
- Created a fresh GCP account — **$300 free credit, valid 90 days**, no charges
  unless you manually upgrade (you won't be billed accidentally)
- Hit the same CPU quota issue initially (12 vCPU default limit)
- Settled on a `c2d-standard-8` instance (8 vCPUs, 32 GB RAM) in
  `europe-west1-b` (Belgium) — fits within the free quota
- OS: Ubuntu 22.04 LTS, 100 GB disk
- Cost: ~$1.61/hr → a 12h overnight run costs ~$19 out of $300 budget

### Setting up the environment
- Installed Docker on the VM
- Pulled the official FEniCSx Docker image (`dolfinx/dolfinx:stable`) —
  this replaces ALL the conda setup we did on the cluster, takes 3 minutes
- Cloned our GitHub repo directly onto the VM
- Hit a Git LFS issue — the `.h5` mesh files were stored as LFS pointers
  (132 bytes instead of hundreds of MB). Fixed by installing `git-lfs` and
  running `git lfs pull`

### Simulation — currently running 🟢
The F-35 Euler simulation (`solver_euler.py`) is currently running on the GCP
VM with 8 MPI ranks on the `lvl2` mesh. As of ~midnight:
- All 8 Python MPI processes are pegged at ~100% CPU
- Each rank using ~1.5 GB RAM (total ~12 GB / 32 GB available)
- No crashes, no memory issues
- Checkpoint saves are configured every 90 minutes so nothing is lost if
  something goes wrong overnight

Expected to finish sometime during the night/early morning.

---

## When you wake up — how to check results

SSH into the VM from your browser (no install needed):

1. Go to https://console.cloud.google.com
2. Sign in with the Google account Wassim used to create the project
3. Hamburger menu (top left) → **Compute Engine** → **VM Instances**
4. Find `f35-euler-sim` → click the **SSH** button on the right
5. A terminal opens in the browser

Then run these commands to check status:

```bash
# Is the sim still running?
ps aux | grep mpirun

# Did it finish? (look for these output files)
ls -lh ~/sim_files/ST7-HPC-Aeronautics-Project/cluster_files/f35_euler/results_f35*

# Check the last lines of output / any error messages
# (only works if sim is still running)
tail -50 /proc/$(pgrep -f solver_euler | head -1)/fd/1 2>/dev/null

# Check for checkpoint files (saved every 90min)
ls -lh ~/sim_files/ST7-HPC-Aeronautics-Project/cluster_files/f35_euler/checkpoint*
```

If `results_f35.xdmf` and `results_f35.h5` exist → **simulation finished successfully**.

---

## Downloading the results

In the GCP SSH browser terminal, there's an **UPLOAD FILE / DOWNLOAD FILE**
button in the top right corner of the SSH window. You can use that to download
result files directly to your laptop.

Or from a local terminal if you have `gcloud` installed:
```bash
gcloud compute scp \
  f35-euler-sim:~/sim_files/ST7-HPC-Aeronautics-Project/cluster_files/f35_euler/results_f35* \
  ./results/ \
  --zone=europe-west1-b
```

---

## What still needs to be done tomorrow

- [ ] Confirm F-35 Euler results look right in ParaView
- [ ] Run F-35 Stokes sim (just change the solver script, same mesh)
- [ ] Get Corsair mesh files ready and repeat the same process
- [ ] If quota increase got approved overnight, switch to `c2d-standard-32`
      (32 cores) for the Corsair sim — much faster

---

## ⚠️ Important — don't forget to stop the VM when done

The VM costs money while running. Once results are downloaded:

1. In the GCP console → Compute Engine → VM Instances
2. Select `f35-euler-sim` → click **Stop**

At ~$1.61/hr we have plenty of budget, but no point burning credits
while the VM sits idle.

---

## Key credentials / access info

- **Platform:** Google Cloud Platform
- **Project:** My First Project (project ID in the console URL)
- **VM name:** `f35-euler-sim`
- **Zone:** `europe-west1-b` (Belgium)
- **Instance type:** `c2d-standard-8` (8 vCPUs, 32 GB RAM)
- **OS:** Ubuntu 22.04 LTS
- **Sim files location on VM:**
  `~/sim_files/ST7-HPC-Aeronautics-Project/cluster_files/f35_euler/`
- **Free credit remaining:** ~€260 out of €300 (shown in the yellow bar
  at the top of the GCP console)

Ask Wassim for the Google account login if you need to access the console.

---

*Good luck everyone — fingers crossed for clean results by morning 🤞*
