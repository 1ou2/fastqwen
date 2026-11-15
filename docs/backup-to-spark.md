# Back up the temporary GPU server to the Spark

Run the backup from your local DGX Spark while the GPU server is still available.
The default SSH alias is `gpu-server`, matching your working `scp` syntax.

## Reported successful transfer: November 15, 2026

The user ran `bash backup_to_spark.sh .` from the Spark checkout's `h100`
directory and supplied output ending with:

```text
Source: gpu-server:/opt/home/gabriel/dev/fastqwen/
Destination: /home/gabriel/dev/fastqwen/h100
Verified backup: /home/gabriel/dev/fastqwen/h100
```

This is a user-reported successful transfer and second checksum comparison for
the script's selected files, not an independent inspection of the Spark copy.
The selected scope includes the trained draft's
[final export](../data/runs/sft-pilot-20261104/final), retained optimizer
checkpoints, and both completed draft comparisons under `data/runs/`.
The final `model.safetensors` on the source is 3,763,692,048 bytes.

The reported 20,560,220 transferred bytes are not the total backup size:
unchanged destination files are skipped. Writer quiescence and a second durable
copy were not established by the supplied output. Rerun the backup after any new
experiments or documentation changes and verify again before releasing the server.

## Scope

The backup preserves the repository-relative directory structure and includes:

- `data/runs/`: all runs, not just the overnight wave; candidates, classification
  decisions and exports, generated answers, manifests, logs, smoke outputs, and
  any other files saved within runs. This automatically includes the completed
  English run, French/code preparation, smoke tests, the November 3-4 bulk run,
  and its sibling launcher log; no per-run script update is needed.
- `benchmarks/`: benchmark runs and their prompts, quality sets, validation
  inputs, exclusion manifest, and tokenizer contract.
- `data/prompts/`: authored prompts, tools, and any local prompt shards.
- `checkpoints/` and `data/checkpoints/`, when present: trained weights, tokenizer
  files, optimizer/scheduler state, and other checkpoint contents required to
  resume training. Checkpoints saved under `data/runs/` are also included.
- `data/calib/`, when present: calibration inputs and their manifests.
- `docs/`: all documentation, including untracked notes and the night analysis.
- Root-level `*.md`: the README and project plan.

It does not copy Git history, application code, Python environments, dependency
caches, Docker images, downloaded base-model weights, or `data/corpora/`. Trained
weights in the checkpoint directories above are included. The downloaded corpora
can be fetched again; the sampled prompts used for runs are preserved. Files
outside this scope, including container logs not saved into a run directory,
are not backed up. Save any additional experiment artifacts into a run directory
before the final transfer. Push code changes to Git separately before losing the
server; this backup is not a substitute for doing so.

## Transfer

Both machines need `rsync`; the Spark also needs Bash, SSH, and standard Linux
utilities. Existing SSH authentication is used; no credentials are embedded.
Copy the script to the Spark, then execute it there:

```bash
scp gpu-server:/opt/home/gabriel/dev/fastqwen/scripts/backup_to_spark.sh .
bash backup_to_spark.sh "$HOME/backups/fastqwen"
```

The completed French/code wave is included automatically when you rerun the
command against the same local destination. Ensure enough free space for the new
outputs and future training checkpoints, which can be much larger than JSONL
datasets. To use another host alias or remote repository path:

```bash
bash backup_to_spark.sh "$HOME/backups/fastqwen" gpu-server /opt/home/gabriel/dev/fastqwen
```

Transfers preserve timestamps and permissions, resume partial files, and never
delete local files. Checksums also detect and repair local corruption even when
file sizes and timestamps match. Rerun the same command after an interrupted
transfer or to collect new runs. Updated remote files overwrite their
corresponding local copies, so keep independent copies of any locally edited
backup files.

## Final verification and recovery

After transfer, the script performs a second, read-only `rsync --checksum` pass.
It succeeds only if all selected remote files match their local counterparts.
Any transfer error or detected difference results in a nonzero exit status.
Checksum verification rereads the selected files on both machines and therefore
takes additional time. Extra local files are retained and are not compared as
remote deletions.

Stop jobs that write to the selected directories before the final backup. A
live transfer is not an atomic snapshot, and checksums cannot freeze files that
continue changing. Do not release the server until this final, quiescent backup
reports `Verified backup`. Keep another copy on durable storage as well.

On the Spark, obtain the code from Git separately. Copy the backed-up directories
into that checkout when needed, without deleting existing files:

```bash
rsync -a "$HOME/backups/fastqwen/" "$HOME/dev/fastqwen/"
```

Install the appropriate Spark dependencies and download model revisions again;
the H100 Docker setup is documented, but not necessarily suitable for Spark.

The overnight analysis is in
[night-run-20261101-analysis.md](night-run-20261101-analysis.md). The French/code
bulk layout is documented in [french-code-step2-bulk.md](french-code-step2-bulk.md).