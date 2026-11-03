# Back up the temporary GPU server to the Spark

Run the backup from your local DGX Spark while the GPU server is still available.
The default SSH alias is `gpu-server`, matching your working `scp` syntax.

## Scope

The backup preserves the repository-relative directory structure and includes:

- `data/runs/`: all runs, not just the overnight wave; candidates, classification
  decisions and exports, generated answers, manifests, logs, smoke outputs, and
  any other files saved within runs.
- `benchmarks/`: benchmark runs and their prompts, quality sets, validation
  inputs, exclusion manifest, and tokenizer contract.
- `data/prompts/`: authored prompts, tools, and any local prompt shards.
- `docs/`: all documentation, including untracked notes and the night analysis.
- Root-level `*.md`: the README and project plan.

It does not copy Git history, application code, Python environments, dependency
caches, Docker images, model weights, or `data/corpora/`. The downloaded corpora
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

The selected data occupied approximately 1.5 GB on November 3, 2026. Ensure enough
free space for later runs. To use another host alias or remote repository path:

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
[night-run-20261101-analysis.md](night-run-20261101-analysis.md).