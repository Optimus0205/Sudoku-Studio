#include <bits/stdc++.h>
using namespace std;

class Solution {
private:
    bool isSafe(int row, int col, int num, vector<vector<int>>& board) {
        for(int i = 0; i < 9; i++) {
            if(board[row][i] == num || board[i][col] == num) {
                return false;
            }
        }

        int startRow = row - row % 3;
        int startCol = col - col % 3;

        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                if(board[startRow + i][startCol + j] == num) {
                    return false;
                }
            }
        }

        return true;
    }

public:
    bool solveSudoku(vector<vector<int>>& board) {
        // write code here 
        int n = board.size();
        int m = board[0].size();

        for(int row = 0; row < n; row++) {
            for(int col = 0; col < m; col++) {
                if(board[row][col] == 0) {
                    for(int num = 1; num <= 9; num++) {
                        if(isSafe(row, col, num, board)) {
                            board[row][col] = num;
                            if(solveSudoku(board)) return true;
                            board[row][col] = 0;
                        }
                    }

                    return false;
                }
            }
        }

        return true;
    }

    void printBoard(vector<vector<int>>& board) {
        for(int row = 0; row < 9; row++) {
            for(int col = 0; col < 9; col++) {
                cout << board[row][col] << " ";
            }

            cout << endl;
        }
    }
};

int main() {
    vector<vector<int>> board = {
        {0,0,2,0,1,0,0,0,0},
        {0,9,0,7,0,6,0,0,0},
        {3,0,0,0,0,0,0,0,4},
        {0,5,0,0,0,1,3,0,0},
        {0,0,0,0,0,7,0,4,9},
        {4,0,0,9,0,0,6,0,0},
        {0,0,1,0,0,4,8,0,3},
        {0,0,3,0,0,0,0,7,0},
        {6,0,7,0,0,0,0,0,0}
    };

    Solution obj;

    if (obj.solveSudoku(board)) {
        cout << "Solved Sudoku:\n";
        obj.printBoard(board);
    } else {
        cout << "No solution exists.\n";
    }

    return 0;
}