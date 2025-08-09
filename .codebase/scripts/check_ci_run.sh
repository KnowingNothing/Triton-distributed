#!/bin/bash
PLATFORMS=("nvidia" "amd")
JOB_TYPES=("unittest" "e2e" "megakernel" "internal" "compile-check")

proceed_jobs=()
skipped_jobs=()

for platform in "${PLATFORMS[@]}"; do
    for job_type in "${JOB_TYPES[@]}"; do
        echo "- Platform: $platform, Job: $job_type"
        export PLATFORM="$platform"
        export JOB_TYPE="$job_type"
        DECISION=$(python3 .codebase/scripts/pre_flight_check.py)
        
        echo "Result: $DECISION"
        
        if [[ "$DECISION" == "PROCEED" ]]; then
            proceed_jobs+=("Platform: $platform, Job: $job_type")
        else
            skipped_jobs+=("Platform: $platform, Job: $job_type")
        fi

        echo "----------------------------------------"
    done
done

echo "=============== Summary ==============="
echo ""

# 打印将要执行的Job列表
echo "✅ Jobs to proceed:"
if [ ${#proceed_jobs[@]} -eq 0 ]; then
    echo "   (None)"
else
    for job in "${proceed_jobs[@]}"; do
        echo "   - $job"
    done
fi

echo ""
echo "❌ Skipped jobs:"
if [ ${#skipped_jobs[@]} -eq 0 ]; then
    echo "   (None)"
else
    for job in "${skipped_jobs[@]}"; do
        echo "   - $job"
    done
fi

echo ""
echo "============================================"
