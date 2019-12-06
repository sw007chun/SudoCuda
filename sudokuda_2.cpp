#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <ctime>
// #include <cuda.h>
using namespace std;

#define N            4
#define BOARD_WIDTH  16
#define BOARD_SIZE   256   //3: 81, 4: 256, 5: 625, 6: 1296, 7: 2401
#define DIFF         14    //3:  8, 4: 14,  5:  22, 6:   32, 7:   44
#define N_ASSIGN     30    //3: 17, 4: 30,  5:  47, 6:   68, 7:   93
#define SET_BIT      0xFFFF   //9 = 0x1FF, 16 = 0xFFFF, 25 = 0x1FFFFFF, 36 = 0xFFFFFFFFF, 49 = 0x1FFFFFFFFFFFF, 64 = 0xFFFFFFFFFFFFFFFF

class Possible {
   uint32_t _b;   //using bitwise 
public:
   Possible() : _b(SET_BIT) {}  
   // check if number i is possible number for the cell
   bool   is_on(int i) const { return ((_b >> (i-1)) & 1); } 
   //removing number i from possible number of the cell
   void   eliminate(int i)   { _b &= ~(uint32_t(1) << (i-1)); }   
   // return first number(from 1) from the cell
   int    val()        const;
   // count number of possible numbers of the cell
   int    count()      const;

   string str(int width) const;
};

int Possible::val() const {
   // _b should not be 0, there should be at least one possible number
   if (_b == 0) return -1;

   uint32_t count = 0, n = _b;
   // shift bits until meet bit 1
   while ( (n & 1) == 0) {
      n >>= 1;
      ++count;
   }
   // 1 should be added because i-th bit represent number i
   return count + 1;
}

int Possible::count() const {
   uint32_t count = 0, n = _b;
   // count all 1 bits
   while (n > 0) {
      count += n & 1;
      n >>= 1;
   }
   return count;
}

// This should actually make string of possible numbers but now it works only for the final outputs
// single quotation marks are added to two digit numbers
string Possible::str(int width) const {
   string s(width, ' ');
   for (int i = 1; i <= BOARD_WIDTH; ++i) 
      if (is_on(i)) s = string(width - to_string(i).size(), ' ') + to_string(i);
   
   return s;
} 

class Sudoku {
   // cell is in 1-d array
   Possible _cells[BOARD_SIZE];
   static uint16_t _group[3*BOARD_WIDTH][BOARD_WIDTH], _neighbors[BOARD_SIZE][3*BOARD_WIDTH-3], _groups_of[BOARD_SIZE][3];

   bool     eliminate(int k, int val);
public:
   Sudoku() {};
   Sudoku(string s);
   static void init();

   // return the possible object of the cell
   Possible possible(int k) const { return _cells[k]; }
   // check if the Sudoku has been solved
   bool     is_solved() const;
   // assign val to _cell[k]
   bool     assign(int k, int val);
   // return the array of cell with least possible numbers left
   void      least_count(int* least_k) const;
   // Print the solved Sudoku board
   void     write(ostream& o) const;
   // checks number of unassigned Sudoku cells
   int      unassigned() const;
};

int Sudoku::unassigned() const {
   int count = 0;
   for (int k = 0; k < BOARD_SIZE; ++k) {
      // cells with count = 1, assigned to 1 number
      if (_cells[k].count() != 1) {
         ++count;
      }
   }
   return count;
}

bool Sudoku::is_solved() const {
   for (int k = 0; k < BOARD_SIZE; ++k) {
      if (_cells[k].count() != 1) {
         return false;
      }
   }
   return true;
}

void Sudoku::write(ostream& o) const {
   int width = to_string(BOARD_WIDTH).size() + 1;
   for (int k = 0; k < BOARD_SIZE; ++k) {
      width = max(width, 1 + _cells[k].count());
   }
   const string sep(N * width + 1, '-');
   for (int i = 0; i < BOARD_WIDTH; ++i) {
      if (i > 0 && i % N == 0) {
         for (int k = 0; k < N - 1; ++k)
            o << sep << '+';
         
         o << sep << endl;
      }
      for (int j = 0; j < BOARD_WIDTH; ++j) {
         if (j > 0 && j % N == 0) o << " |";
         o << _cells[i*BOARD_WIDTH + j].str(width);
      }
      o << endl;
   }
   o << endl;
}

uint16_t
Sudoku::_group[3*BOARD_WIDTH][BOARD_WIDTH] = {}, Sudoku::_neighbors[BOARD_SIZE][3*BOARD_WIDTH-3]={}, Sudoku::_groups_of[BOARD_SIZE][3]={};
// initializing static valiables, which designates gruop numbers and neighbors of a cell
// _group : total 3 * BOARD_WIDTH, 1 for row group, 2 for column group, 1 for square group
// _groups_of[k] : what group does a _cell[k] belongs to. Each cell belong to 3 groups
// _neighbors[k] : neighbors of _cell[k] excluding itself
void Sudoku::init() {
   for (int i = 0; i < BOARD_WIDTH; ++i) {
      for (int j = 0; j < BOARD_WIDTH; ++j) {
         const int k = i*BOARD_WIDTH + j;
         //Which group does k belongs to, x[0]: row group, x[1]: column gorup, x[2]: square group
         const int x[3] = {i, BOARD_WIDTH + j, 2*BOARD_WIDTH + (i/N)*N + j/N};

         _group[x[0]][j] = k;
         _group[x[1]][i] = k;
         _group[x[2]][(i%N)*N + (j%N)] = k;

         for (int g = 0; g < 3; g++) 
            _groups_of[k][g] = x[g];
      }
   }

   for (int k = 0; k < BOARD_SIZE; ++k) {
      int l = 0;
      for (int x = 0; x < 3; x++) {
         for (int j = 0; j < BOARD_WIDTH; ++j) {
            int k2 = _group[_groups_of[k][x]][j];
            if (k2 != k) _neighbors[k][l++] = k2;
         }
      }
   }
}

// assign _cell[k] = val -> remove all other values from cell k and its neighbors
bool Sudoku::assign(int k, int val) {
   for (int i = 1; i <= BOARD_WIDTH; ++i) {
      if (i != val) {
         //eliminate other values and if not possible wrong approach
         if (!eliminate(k, i)) return false;
      }
   }
   return true;
}

bool Sudoku::eliminate(int k, int val) {
   // if val is already eliminated in _cell[k] return
   if (!_cells[k].is_on(val)) {
      return true;
   }

   _cells[k].eliminate(val);

   // count number of all posible values left for cell[k] after removing val
   const int n = _cells[k].count();
   // N should be at least 1, it is a conflict if n is 0
   if (n == 0) {
      return false;
   // if there is only one possible value left for cell[k] assigned that value to the _cell[k]
   // and eliminate that number from neighbors 
   } else if (n == 1) {
      // find that unique value v
      const int v = _cells[k].val();
      // remove v from all the neighbors
      for (int i = 0; i < 3*BOARD_WIDTH-3; ++i) {
         if (!eliminate(_neighbors[k][i], v)) return false;
      }
   }

   // Check count of possible numbers of the neighbors to check conflict or assign a number
   // there are three possible group for each cell[k]
   for (int i = 0; i < 3; ++i) {
      // get the group number x
      const int x = _groups_of[k][i];
      int n = 0, ks;
      // for each neighbors in the group x
      for (int j = 0; j < BOARD_WIDTH; ++j) {
         const int p = _group[x][j];
         // count the number of possible values for each neighbor in the i-th group
         if (_cells[p].is_on(val)) {
            n++, ks = p;
         }
      }
      // it's a conflict if n = 0 for a cell
      if (n == 0) {
         return false;
      // if there is only one possible value, assign such value. Constraint propagation
      } else if (n == 1) {
         if (!assign(ks, val)) {
            return false;
         }
      }
   }
   return true;
}

// returns cell number k with least possible numbers left
void Sudoku::least_count(int* least_k) const {
   int k = -1, min;
   // for all the cells
   for (int i = 0; i < BOARD_SIZE; ++i) {
      // get the count of possible numbers a cell can take
      const int m = _cells[i].count();
      // update minimum
      if (m > 1 && (k == -1 || m < min)) {
         min = m, k = i;
      }
   }

   int l = 0;
   for (int i = 0; i < BOARD_SIZE; ++i) {
      const int m = _cells[i].count();
      // add the cell number with least possible numbers left
      if (m == min) {
         least_k[l++] = i;
      }
   }
   // return least_k;
}

// Parse Sudoku board from a string
Sudoku::Sudoku(string s) 
{
   int k = 0;
   for (int i = 0; i < s.size(); ++i) {
      // if there is a single quotation mark it's 2 digit number
      if (s[i] == '\'') {
         int num = stoi(s.substr(i+1, i+3));
         if (!assign(k,num)) {
            cerr << "error" << endl;
            return;
         }
         ++k;
         i += 3;
      }
      else if (s[i] >= '1' && s[i] <= '9') {
         if (!assign(k, s[i] - '0')) {
            cerr << "error" << endl;
            return;
         }
         ++k;
      } 
      // 0 or . can be used to unassigned cells
      else if (s[i] == '0' || s[i] == '.') {
         ++k;
      }
   }
}

vector<Sudoku*> solutions;

Sudoku* solve(Sudoku* S, int branches, int depth) {
   ++depth;
   // current branch depth and number of unassigned cells
   cout << "depth : " << depth << ", unassigned : " << S->unassigned() <<  endl;
   // base case
   if (S == nullptr || S->is_solved()) {
      solutions.push_back(S);
      return S;
   }
   // cell with least possible #
   int least_k[BOARD_SIZE];
   S->least_count(least_k);
   // branches += least_k.size();
   // cout << branches << endl;

   int loop = 1;
// this code was used to branch out at the first depth only
   // if (branches == 1) {
   //    loop = least_k.size();
   // }

   for (int j = 0; j < loop; ++j) {
      int k = least_k[j];      
      // possible number of that cell
      Possible p = S->possible(k);
      for (int i = 1; i <= BOARD_WIDTH; ++i) {
         // if number i is possible on _cell[k]
         if (p.is_on(i)) {
            // make new Sudokue board
            Sudoku* S1 = new Sudoku(*S);
            // if there is no conflict in assign number i to cell[k] go deeper
            if (S1->assign(k, i)) {
               if (auto S2 = solve(S1, branches + loop, depth)) {
                  if (branches != 1)
                     return S2;
                  else 
                     break;
               }
            }
            delete S1;
         }
      }
   }
   return {};
}

// Random Sudoku generator shuffles the Sudoku cells randomly and assign random number to cells
// N_ASSIGN : number of cells to be assigne
// DIFF : number of assigned unique numbers
string rand_generator() {
   int val;
   int cell_num[BOARD_SIZE];
   string cell_val[BOARD_SIZE];
   bool err = false;

   for (int i = 0; i < BOARD_SIZE; ++i) 
      cell_num[i] = i;
   
   // count unassigned numbers, at least 8 different numbers should be assigned to 9x9 Sudoku
   Possible* unassigned_nums = new Possible();
   int assigned_num[N_ASSIGN];

   srand(time(0));

   // shuffle cell numbers to assign values
   shuffle(cell_num, cell_num + BOARD_SIZE, default_random_engine(time(0)));

   // should have at least 8 different number for 9x9 Sudoku.(9 - unassigned = 8)
   while( BOARD_WIDTH - unassigned_nums->count() < DIFF) {
      Sudoku s;
      // assign at least N_ASSIGN(=17) numbers for 9x9 Sudoku
      for (int i = 0; i < N_ASSIGN; ++i) {
         val = rand() % (BOARD_WIDTH) + 1;
         // cout << val << endl;
         // elimninate assigned value
         unassigned_nums->eliminate(val);
         assigned_num[i] = val;
         // check if there is any conflict
         if (!s.assign(cell_num[i], val)) {
            // cerr << "error" << endl;
            err = true;
            continue;
         }
      }
      // if error occured make new object and start again
      if (err) {
         delete unassigned_nums;
         unassigned_nums = new Possible();
         err = false;
      }
   }
   delete unassigned_nums;
   // delete s;

   for (int i = 0; i < BOARD_SIZE; ++i) {
      cell_val[i] = ".";
   }

   for (int i = 0; i < N_ASSIGN; ++i) {
      if (assigned_num[i] > 9)
         cell_val[cell_num[i]] = '\'' + to_string(assigned_num[i]) + '\'';
      else
         cell_val[cell_num[i]] = to_string(assigned_num[i]);
   }

   string ret = "";
   for (int i = 0; i < BOARD_SIZE; ++i) {
      ret += cell_val[i];
   }
   return ret;
}

int main() {
   Sudoku::init();
   string line = rand_generator();
   cout << line << endl;
   while (getline(cin, line)) {
      if (auto S = solve(new Sudoku(line), 1,0)) {
         S->write(cout);
      } else {
         cout << "No solution";
      }
      cout << endl;

      for (int i = 0; i < solutions.size(); ++i) {
         solutions[i]->write(cout);
      }
   }
}
