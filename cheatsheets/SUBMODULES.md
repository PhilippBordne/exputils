## Working with submodules
```bash
### Adding
git submodule add <url> externals/<name>      # add a submodule at a given path
git submodule add -b <branch> <url> <path>   # track a specific branch

### Cloning a repo that has submodules
git clone --recurse-submodules <url>         # clone and populate submodules in one step
git submodule update --init --recursive      # populate submodules in an already-cloned repo

### Updating
git submodule update --remote <path>         # fetch + check out latest from tracked branch
git submodule update --init --recursive      # check out the commits currently pinned by the parent repo

### Pinning a new commit (inside the submodule)
cd externals/<name>
git checkout <commit-or-branch>              # move submodule to desired commit
cd ../..
git add externals/<name>                      # stage the new pin in the parent repo
git commit -m "Bump <name> to <commit>"      # record it

### Inspection
git submodule status                         # show each submodule's pinned commit
git submodule foreach <cmd>                  # run a command in every submodule

### Removing
git submodule deinit <path>                  # unregister (clears from .git/config + working tree)
git rm <path>                                # remove the submodule entry + .gitmodules line
```

## Working with forks
How to keep the fork up-to-date with the original repository (upstream).
```bash
git remote add upstream <original-repo-url>   # one-time: register the source you forked from
git fetch upstream                            # get its latest commits
git checkout main                             # your fork's branch
git merge upstream/main                       # or: git rebase upstream/main
git push origin main                          # update your fork on the host (optional)
```
Alternatively the fork can be synced on GitHub via "Sync fork".