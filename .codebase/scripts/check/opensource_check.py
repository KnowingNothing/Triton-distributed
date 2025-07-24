# -*- coding: UTF-8 -*-
import re
import os
import sys
import subprocess
import argparse
from typing import List
import fnmatch
import logging
from pathlib import Path

GIT_COMMIT_LOG = "/tmp/opensource_git_commit.log"


# 获取commit log
def opensource_get_commit_log(repo_folder_path, depth=None):
    os.chdir(repo_folder_path)
    depth_options = f"--max-count={depth}" if depth else ""
    # -p output the patch introduced by each commit
    subprocess.run(
        f"git log {depth_options} > {GIT_COMMIT_LOG}",
        shell=True,
        stdout=subprocess.PIPE,
    )


def should_ignore_keywords(ignore_list_keywords, line):
    for ignore_key in ignore_list_keywords:
        pattern1 = re.compile(ignore_key, flags=re.I)
        if len(pattern1.findall(line)) > 0:
            return True
    return False


def is_exclude_path(exclude_patterns: List[str], path):
    return any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns)


def get_git_ls_files(repo_folder_path):
    os.chdir(repo_folder_path)
    result = subprocess.run("git ls-files", shell=True, stdout=subprocess.PIPE, text=True)
    files = result.stdout.splitlines()
    return files


def check_sensitive_information(
    keywords_list,
    ignore_list_keywords,
    repo_folder_path,
    exclude_patterns: List[str],
    cnt,
    aigc_keywords_group1,
    aigc_keywords_group2,
    skip_git_commit_log: bool = False,
):
    has_sensitive_info = False
    extra_path = [] if skip_git_commit_log else [GIT_COMMIT_LOG]
    for file_path in get_git_ls_files(repo_folder_path) + extra_path:
        file_path = os.path.join(repo_folder_path, file_path)
        logging.debug("Checking file: %s", os.path.relpath(file_path, repo_folder_path))
        if not Path(file_path).exists() or not Path(file_path).is_file() or is_exclude_path(
                exclude_patterns, os.path.relpath(file_path, repo_folder_path)):
            logging.info(f"Skipping excluded file: {file_path}")
            continue
        if file_path.endswith((
                ".tgz",
                ".zip",
                ".tar",
                ".rar",
                ".gif",
                ".jpg",
                ".png",
                ".jpeg",
                ".svg",
                ".tiff",
                ".raw",
                ".ico",
                ".webp",
                ".tga",
        )) or (file_path.find("/.git/") > -1):
            continue
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            for line in f.readlines():
                cnt += 1
                if not should_ignore_keywords(ignore_list_keywords, line.strip()):
                    for pattern in keywords_list:
                        result = re.search(pattern, line.strip(), re.I)
                        if (pattern == "(tokenizer|transformer|token_id|tokenid|attention_head).{0,20}"):
                            if file_path.endswith((".json")):
                                if result is not None:
                                    print('\033[1;36mFile "' + file_path + ", line, " + str(cnt) + ',"\033[0m' +
                                          " have some sensitive information: " + "\033[1;32m" + str(result.group(0)) +
                                          "\033[0m")
                                    has_sensitive_info = True
                                else:
                                    continue
                        else:
                            if result is not None:
                                print('\033[1;36mFile "' + file_path + ", line, " + str(cnt) + ',"\033[0m' +
                                      " have some sensitive information: " + "\033[1;32m" + str(result.group(0)) +
                                      "\033[0m")
                                has_sensitive_info = True
                            else:
                                continue
        cnt = 0

        with open(file_path, "r", encoding="ISO-8859-1") as g:
            if str(file_path).endswith(".json"):
                file_content = g.read()
                # 检查每组关键词是否都在文件内容中
                if all(keyword in file_content for keyword in aigc_keywords_group1):
                    print('\033[1;36mFile "' + file_path + " have some aigc sensitive information: " + "\033[1;32m" +
                          str(aigc_keywords_group1) + "\033[0m")
                    has_sensitive_info = True
                elif all(keyword in file_content for keyword in aigc_keywords_group2):
                    print('\033[1;36mFile "' + file_path + " have some aigc sensitive information: " + "\033[1;32m" +
                          str(aigc_keywords_group2) + "\033[0m")
                    has_sensitive_info = True
            else:
                continue

    return has_sensitive_info


def _parse_args():
    parser = argparse.ArgumentParser(description="This script checks for sensitive information in the codebase.")
    parser.add_argument(
        "--repo_folder_path",
        type=str,
        required=True,
        help="Path to the repository folder where the codebase is located.",
    )
    parser.add_argument("--exclude", default=None, help="exclude path to skip")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--depth", type=int, default=None, help="Limit the git log to a specific depth.")
    parser.add_argument("--skip_git_commit_log", default=False, action="store_true")
    return parser.parse_args()


def _load_gitignore(repo_folder_path):
    gitignore_path = os.path.join(repo_folder_path, ".gitignore")
    exclude_patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Convert gitignore patterns to fnmatch patterns
                    if line.endswith("/"):
                        exclude_patterns.append(line + "*")
                    else:
                        exclude_patterns.append(line)
    return exclude_patterns


def main():
    args = _parse_args()
    cnt = 0
    repo_folder_path = args.repo_folder_path
    exclude_patterns = list(filter(None, args.exclude.split(","))) + _load_gitignore(repo_folder_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if not args.skip_git_commit_log:
        opensource_get_commit_log(repo_folder_path, args.depth)
    keywords_list = [
        # r"[a-zA-z0-9]{8}(-[a-zA-z0-9]{4}){3}-[a-zA-z0-9]{12}", # DON'T CHECK UUID
        r"npm\s{1,20}install.{1,30}",
        r"AKLT\w{40,70}",
        r"AKAP\w{40,70}",
        r"(tokenizer|transformer|token_id|tokenid|attention_head).{0,20}",
        r"(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
        r"(LTAI)[a-z0-9]{20}",
        r"AKTP\w{40,70}",
        r"([^*<\s|:>]{0,7})(app_id|appid)([^]()!<>;/@&,]{0,10}[(=:]\s{0,6}[\"']{0,1}[0-9]{6,32}[\"']{0,1})",
        r".{0,15}\.?byted.org.{0,20}",
        r".{0,15}\.?bytedance.net.{0,20}",
        r".{0,20}.bytedance\.feishu\.cn.{0,50}",
        r".{0,20}.bytedance\.larkoffice\.com.{0,50}",
        r"(10\.\d{1,3}\.\d{1,3}\.\d{1,3})",
        r"([^*<\s|:>]{0,4})(testak|testsk|ak|sk|key|token|auth|pass|cookie|session|password|app_id|appid|secret_key|access_key|secretkey|accesskey|credential|secret|access)(\s{0,10}[(=:]\s{0,6}[\"']{0,1}(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{16,32}[\"']{0,1})",
    ]
    aigc_keywords_group1 = ["token", "temp", "role"]
    aigc_keywords_group2 = ["layer", "token", "head"]
    ignore_list_keywords = [
        r"[^*<>]{0,6}token[^]()!<>;/@&,]{0,10}[=:].{0,1}null,",
        r".{0,5}user.{0,10}[=:].{ 0,1}null",
        r".{0,5}pass.{0,10}[=:].{0,1}null",
        r"passport[=:].",
        r"[^*<>]{0,6}key[^]()!<>;/]{0,10}[=:].{0,1}string.{0,10}",
        r".{0,5}user.{0,10}[=:].{0,1}string",
        r".{0,5}pass.{0,10}[=:].{0,1}string",
        r".{0,5}app_id[^]()!<>;/@&,]{0,10}[=:].{0,10}\+",
        r".{0,5}appid[^]()!<>;/@&,]{0,10}[=:].{0,10}\+",
    ]

    has_sensitive_info = check_sensitive_information(
        keywords_list,
        ignore_list_keywords,
        repo_folder_path,
        exclude_patterns,
        cnt,
        aigc_keywords_group1,
        aigc_keywords_group2,
        args.skip_git_commit_log,
    )
    if has_sensitive_info:
        logging.fatal("Sensitive information found in the codebase. Please check the logs.")
        sys.exit(1)
    else:
        print("No sensitive information found in the codebase.")

    if not args.skip_git_commit_log:
        Path(GIT_COMMIT_LOG).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
