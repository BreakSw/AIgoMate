"""Run equivalent Python, Java, and C++ 3Sum programs with Judge0.

Each program contains the same four-case test harness. The SDK sends the three
programs as a batch, which consumes three Judge0 submissions in total.
"""

from __future__ import annotations

import json
import sys

import judge0


EXPECTED_STDOUT = """\
case 1: PASS
case 2: PASS
case 3: PASS
case 4: PASS
ALL_TESTS_PASSED 4/4
"""


PYTHON_SOURCE = r"""
class Solution:
    def threeSum(self, nums):
        nums.sort()
        answer = []

        for first in range(len(nums) - 2):
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            if nums[first] > 0:
                break

            left, right = first + 1, len(nums) - 1
            while left < right:
                total = nums[first] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    answer.append([nums[first], nums[left], nums[right]])
                    left_value, right_value = nums[left], nums[right]
                    while left < right and nums[left] == left_value:
                        left += 1
                    while left < right and nums[right] == right_value:
                        right -= 1
        return answer


def normalize(triples):
    return sorted(sorted(triple) for triple in triples)


def run_case(number, values, expected):
    actual = Solution().threeSum(values)
    passed = normalize(actual) == normalize(expected)
    print(f"case {number}: {'PASS' if passed else 'FAIL'}")
    return passed


passed = sum([
    run_case(1, [-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
    run_case(2, [0, 1, 1], []),
    run_case(3, [0, 0, 0], [[0, 0, 0]]),
    run_case(4, [-2, 0, 0, 2, 2], [[-2, 0, 2]]),
])

if passed == 4:
    print("ALL_TESTS_PASSED 4/4")
else:
    print(f"TESTS_FAILED {passed}/4")
    raise SystemExit(1)
"""


JAVA_SOURCE = r"""
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class Main {
    static class Solution {
        public List<List<Integer>> threeSum(int[] nums) {
            Arrays.sort(nums);
            List<List<Integer>> answer = new ArrayList<>();

            for (int first = 0; first + 2 < nums.length; first++) {
                if (first > 0 && nums[first] == nums[first - 1]) {
                    continue;
                }
                if (nums[first] > 0) {
                    break;
                }

                int left = first + 1;
                int right = nums.length - 1;
                while (left < right) {
                    long sum = (long) nums[first] + nums[left] + nums[right];
                    if (sum < 0) {
                        left++;
                    } else if (sum > 0) {
                        right--;
                    } else {
                        answer.add(Arrays.asList(nums[first], nums[left], nums[right]));
                        int leftValue = nums[left];
                        int rightValue = nums[right];
                        while (left < right && nums[left] == leftValue) {
                            left++;
                        }
                        while (left < right && nums[right] == rightValue) {
                            right--;
                        }
                    }
                }
            }
            return answer;
        }
    }

    static void normalize(List<List<Integer>> triples) {
        for (List<Integer> triple : triples) {
            triple.sort(Integer::compareTo);
        }
        triples.sort(Comparator
            .comparingInt((List<Integer> triple) -> triple.get(0))
            .thenComparingInt(triple -> triple.get(1))
            .thenComparingInt(triple -> triple.get(2)));
    }

    static List<List<Integer>> expected(int[][] triples) {
        List<List<Integer>> result = new ArrayList<>();
        for (int[] triple : triples) {
            result.add(new ArrayList<>(Arrays.asList(triple[0], triple[1], triple[2])));
        }
        return result;
    }

    static boolean runCase(int number, int[] input, int[][] expectedTriples) {
        List<List<Integer>> actual = new Solution().threeSum(input);
        List<List<Integer>> wanted = expected(expectedTriples);
        normalize(actual);
        normalize(wanted);
        boolean passed = actual.equals(wanted);
        System.out.println("case " + number + ": " + (passed ? "PASS" : "FAIL"));
        return passed;
    }

    public static void main(String[] args) {
        int passed = 0;
        passed += runCase(1, new int[]{-1, 0, 1, 2, -1, -4},
            new int[][]{{-1, -1, 2}, {-1, 0, 1}}) ? 1 : 0;
        passed += runCase(2, new int[]{0, 1, 1}, new int[][]{}) ? 1 : 0;
        passed += runCase(3, new int[]{0, 0, 0}, new int[][]{{0, 0, 0}}) ? 1 : 0;
        passed += runCase(4, new int[]{-2, 0, 0, 2, 2},
            new int[][]{{-2, 0, 2}}) ? 1 : 0;

        if (passed == 4) {
            System.out.println("ALL_TESTS_PASSED 4/4");
        } else {
            System.out.println("TESTS_FAILED " + passed + "/4");
            System.exit(1);
        }
    }
}
"""


CPP_SOURCE = r"""
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> answer;

        for (int first = 0; first + 2 < static_cast<int>(nums.size()); ++first) {
            if (first > 0 && nums[first] == nums[first - 1]) continue;
            if (nums[first] > 0) break;

            int left = first + 1;
            int right = static_cast<int>(nums.size()) - 1;
            while (left < right) {
                long long sum = static_cast<long long>(nums[first])
                    + nums[left] + nums[right];
                if (sum < 0) {
                    ++left;
                } else if (sum > 0) {
                    --right;
                } else {
                    answer.push_back({nums[first], nums[left], nums[right]});
                    int leftValue = nums[left];
                    int rightValue = nums[right];
                    while (left < right && nums[left] == leftValue) ++left;
                    while (left < right && nums[right] == rightValue) --right;
                }
            }
        }
        return answer;
    }
};

static void normalize(vector<vector<int>>& triples) {
    for (auto& triple : triples) sort(triple.begin(), triple.end());
    sort(triples.begin(), triples.end());
}

static bool runCase(int number, vector<int> input, vector<vector<int>> expected) {
    vector<vector<int>> actual = Solution().threeSum(input);
    normalize(actual);
    normalize(expected);
    bool passed = actual == expected;
    cout << "case " << number << ": " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

int main() {
    int passed = 0;
    passed += runCase(1, {-1, 0, 1, 2, -1, -4}, {{-1, -1, 2}, {-1, 0, 1}});
    passed += runCase(2, {0, 1, 1}, {});
    passed += runCase(3, {0, 0, 0}, {{0, 0, 0}});
    passed += runCase(4, {-2, 0, 0, 2, 2}, {{-2, 0, 2}});

    if (passed == 4) {
        cout << "ALL_TESTS_PASSED 4/4\n";
        return 0;
    }
    cout << "TESTS_FAILED " << passed << "/4\n";
    return 1;
}
"""


PROGRAMS = [
    ("Python", judge0.PYTHON, PYTHON_SOURCE),
    ("Java", judge0.JAVA, JAVA_SOURCE),
    ("C++", judge0.CPP, CPP_SOURCE),
]


def main() -> int:
    submissions = [
        judge0.Submission(
            source_code=source,
            language=language,
            expected_output=EXPECTED_STDOUT,
            cpu_time_limit=3,
            wall_time_limit=10,
            memory_limit=256_000,
            enable_network=False,
        )
        for _, language, source in PROGRAMS
    ]

    try:
        results = judge0.run(submissions=submissions)
    except Exception as error:
        print(json.dumps({
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }, ensure_ascii=False, indent=2))
        return 1

    reports = []
    for (name, _, _), result in zip(PROGRAMS, results):
        actual_stdout = result.stdout or ""
        status = str(result.status) if result.status is not None else "Unknown"
        passed = status == "Accepted" and actual_stdout == EXPECTED_STDOUT
        reports.append({
            "language": name,
            "ok": passed,
            "status": status,
            "actual_stdout": actual_stdout,
            "compile_output": result.compile_output,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "time_seconds": result.time,
            "memory_kb": result.memory,
            "submission_token": str(result.token) if result.token is not None else None,
        })

    all_passed = len(reports) == len(PROGRAMS) and all(item["ok"] for item in reports)
    print(json.dumps({
        "ok": all_passed,
        "sdk_version": judge0.__version__,
        "algorithm": "3Sum (sorting + two pointers)",
        "test_cases_per_language": 4,
        "submission_count": len(submissions),
        "expected_stdout": EXPECTED_STDOUT,
        "results": reports,
    }, ensure_ascii=False, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
