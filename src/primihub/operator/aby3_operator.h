
#ifndef SRC_primihub_operator_ABY3_operator_H
#define SRC_primihub_operator_ABY3_operator_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <unistd.h>
#include <vector>

#include "src/primihub/common/defines.h"
#include "src/primihub/common/type/fixed_point.h"
#include "src/primihub/common/type/type.h"
#include "src/primihub/primitive/ppa/kogge_stone.h"
#include "src/primihub/protocol/aby3/encryptor.h"
#include "src/primihub/protocol/aby3/evaluator/binary_evaluator.h"
#include "src/primihub/protocol/aby3/evaluator/evaluator.h"
#include "src/primihub/protocol/aby3/evaluator/piecewise.h"
#include "src/primihub/protocol/aby3/sh3_gen.h"
#include "src/primihub/util/crypto/prng.h"
#include "src/primihub/util/network/socket/channel.h"
#include "src/primihub/util/network/socket/session.h"

#include "src/primihub/protocol/aby3/runtime.h"
#include "src/primihub/util/network/socket/commpkg.h"
#include "src/primihub/util/network/socket/ioservice.h"
#include "src/primihub/util/network/socket/session.h"

#include "src/primihub/util/eigen_util.h"
#include "src/primihub/util/log.h"
namespace primihub {
const uint8_t VAL_BITCOUNT = 64;
class MPCOperator {
public:
  Channel mNext, mPrev;
  IOService ios;
  Sh3Encryptor enc;
  Sh3BinaryEvaluator binEval;
  Sh3ShareGen gen;
  Sh3Piecewise mdivision;
  Sh3Piecewise mAbs;
  Sh3Piecewise mQuoDertermine;

  Sh3Evaluator eval;
  Sh3Runtime runtime;
  u64 partyIdx;
  string next_name;
  string prev_name;

  MPCOperator(u64 partyIdx_, string NextName, string PrevName)
      : partyIdx(partyIdx_), next_name(NextName), prev_name(PrevName) {}

  int setup(std::string next_ip, std::string prev_ip, u32 next_port,
            u32 prev_port);
  void fini();
  template <Decimal D>
  void createShares(const eMatrix<double> &vals, sf64Matrix<D> &sharedMatrix) {
    f64Matrix<D> fixedMatrix(vals.rows(), vals.cols());
    for (u64 i = 0; i < vals.size(); ++i)
      fixedMatrix(i) = vals(i);
    enc.localFixedMatrix(runtime, fixedMatrix, sharedMatrix).get();
  }

  template <Decimal D> void createShares(sf64Matrix<D> &sharedMatrix) {
    enc.remoteFixedMatrix(runtime, sharedMatrix).get();
  }

  template <Decimal D>
  sf64Matrix<D> createSharesByShape(const eMatrix<double> &val) {
    f64Matrix<D> v2(val.rows(), val.cols());
    for (u64 i = 0; i < val.size(); ++i)
      v2(i) = val(i);
    return createSharesByShape(v2);
  }

  template <Decimal D>
  sf64Matrix<D> createSharesByShape(const f64Matrix<D> &val) {
    std::array<u64, 2> size{val.rows(), val.cols()};
    mNext.asyncSendCopy(size);
    mPrev.asyncSendCopy(size);
    sf64Matrix<D> dest(size[0], size[1]);
    enc.localFixedMatrix(runtime, val, dest).get();
    return dest;
  }

  template <Decimal D> sf64Matrix<D> createSharesByShape(u64 pIdx) {
    std::array<u64, 2> size;
    if (pIdx == (partyIdx + 1) % 3)
      mNext.recv(size);
    else if (pIdx == (partyIdx + 2) % 3)
      mPrev.recv(size);
    else
      throw RTE_LOC;
    sf64Matrix<D> dest(size[0], size[1]);
    enc.remoteFixedMatrix(runtime, dest).get();
    return dest;
  }

  void createShares(const i64Matrix &vals, si64Matrix &sharedMatrix);

  void createShares(int vals, si64 &sharedInt);

  void createShares(si64 &shared);

  void createShares(si64Matrix &sharedMatrix);

  si64Matrix createSharesByShape(const i64Matrix &val);

  si64Matrix createSharesByShape(u64 partyIdx);

  template <Decimal D>
  void createShares(const f64Matrix<D> &vals, sf64Matrix<D> &sharedMatrix) {
    enc.localFixedMatrix<D>(runtime, vals, sharedMatrix).get();
  }

  template <Decimal D> void createShares(double vals, sf64<D> &sharedFixedInt) {
    enc.localFixed<D>(runtime, vals, sharedFixedInt).get();
  }

  template <Decimal D> void createShares(sf64<D> &sharedFixedInt) {
    enc.remoteFixed<D>(runtime, sharedFixedInt).get();
  }

  sbMatrix createBinSharesByShape(i64Matrix &val, u64 bitCount);

  sbMatrix createBinSharesByShape(u64 partyIdx);

  template <Decimal D> eMatrix<double> revealAll(const sf64Matrix<D> &vals) {
    f64Matrix<D> temp(vals.rows(), vals.cols());
    enc.revealAll(runtime, vals, temp).get();

    eMatrix<double> ret(vals.rows(), vals.cols());
    for (u64 i = 0; i < ret.size(); ++i)
      ret(i) = static_cast<double>(temp(i));
    return ret;
  }

  i64Matrix revealAll(const si64Matrix &vals);

  template <Decimal D> double revealAll(const sf64<D> &vals) {
    f64<D> ret;
    enc.revealAll(runtime, vals, ret).get();
    return static_cast<double>(ret);
  }

  template <Decimal D> eMatrix<double> reveal(const sf64Matrix<D> &vals) {
    f64Matrix<D> temp(vals.rows(), vals.cols());
    enc.reveal(runtime, vals, temp).get();
    eMatrix<double> ret(vals.rows(), vals.cols());
    for (u64 i = 0; i < ret.size(); ++i)
      ret(i) = static_cast<double>(temp(i));
    return ret;
  }

  template <Decimal D> void reveal(const sf64Matrix<D> &vals, u64 Idx) {
    enc.reveal(runtime, Idx, vals).get();
  }

  i64Matrix reveal(const si64Matrix &vals);

  void reveal(const si64Matrix &vals, u64 Idx);

  template <Decimal D> double reveal(const sf64<D> &vals) {
    f64<D> ret;
    enc.reveal(runtime, vals, ret).get();
    return static_cast<double>(ret);
  }

  template <Decimal D> void reveal(const sf64<D> &vals, u64 Idx) {
    enc.reveal(runtime, Idx, vals).get();
  }

  // Reveal binary share to one or more party.
  i64Matrix reveal(const sbMatrix &sh_res);
  void reveal(const sbMatrix &sh_res, uint64_t party_id);

  template <Decimal D> sf64<D> MPC_Add(std::vector<sf64<D>> sharedFixedInt) {
    sf64<D> sum;
    sum = sharedFixedInt[0];
    for (u64 i = 1; i < sharedFixedInt.size(); i++) {
      sum = sum + sharedFixedInt[i];
    }
    return sum;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Add(std::vector<sf64Matrix<D>> sharedFixedInt) {
    sf64Matrix<D> sum;
    sum = sharedFixedInt[0];
    for (u64 i = 1; i < sharedFixedInt.size(); i++) {
      sum = sum + sharedFixedInt[i];
    }
    return sum;
  }

  si64Matrix MPC_Add(std::vector<si64Matrix> sharedInt);

  template <Decimal D>
  sf64<D> MPC_Add_Const(f64<D> constfixed, sf64<D> &sharedFixed) {
    sf64<D> temp = sharedFixed;
    if (partyIdx == 0)
      temp[0] = sharedFixed[0] + constfixed.mValue;
    else if (partyIdx == 1)
      temp[1] = sharedFixed[1] + constfixed.mValue;
    return temp;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Add_Const(f64<D> constfixed, sf64Matrix<D> &sharedFixed) {
    sf64Matrix<D> temp = sharedFixed;
    if (partyIdx == 0)
      for (i64 i = 0; i < sharedFixed.rows(); i++)
        for (i64 j = 0; j < sharedFixed.cols(); j++)
          temp[0](i, j) = sharedFixed[0](i, j) + constfixed.mValue;
    else if (partyIdx == 1)
      for (i64 i = 0; i < sharedFixed.rows(); i++)
        for (i64 j = 0; j < sharedFixed.cols(); j++) {
          temp[1](i, j) = sharedFixed[1](i, j) + constfixed.mValue;
        }
    return temp;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Add_Const(f64Matrix<D> constFixedMatrix,
                              sf64Matrix<D> &sharedFixed) {
    auto b0 = sharedFixed[0].cols() != constFixedMatrix.cols();
    auto b1 = sharedFixed[0].rows() != constFixedMatrix.rows();
    if (b0 || b1)
      throw std::runtime_error(LOCATION);
    sf64Matrix<D> temp = sharedFixed;
    if (partyIdx == 0)
      temp[0] = sharedFixed[0] + constFixedMatrix.i64Cast();
    else if (partyIdx == 1)
      temp[1] = sharedFixed[1] + constFixedMatrix.i64Cast();
    return temp;
  }

  si64 MPC_Add_Const(i64 constInt, si64 &sharedInt);

  si64Matrix MPC_Add_Const(i64 constInt, si64Matrix &sharedIntMatrix);

  template <Decimal D>
  sf64<D> MPC_Sub(sf64<D> minuend, std::vector<sf64<D>> subtrahends) {
    sf64<D> difference;
    difference = minuend;
    for (u64 i = 0; i < subtrahends.size(); i++) {
      difference = difference - subtrahends[i];
    }
    return difference;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Sub(sf64Matrix<D> minuend,
                        std::vector<sf64Matrix<D>> subtrahends) {
    sf64Matrix<D> difference;
    difference = minuend;
    for (u64 i = 0; i < subtrahends.size(); i++) {
      difference = difference - subtrahends[i];
    }
    return difference;
  }

  si64Matrix MPC_Sub(si64Matrix minuend, std::vector<si64Matrix> subtrahends);

  template <Decimal D>
  sf64<D> MPC_Sub_Const(f64<D> constfixed, sf64<D> &sharedFixed, bool mode) {
    sf64<D> temp = sharedFixed;
    if (partyIdx == 0)
      temp[0] = sharedFixed[0] - constfixed.mValue;
    else if (partyIdx == 1)
      temp[1] = sharedFixed[1] - constfixed.mValue;
    if (mode != true) {
      temp[0] = -temp[0];
      temp[1] = -temp[1];
    }
    return temp;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Sub_Const(f64<D> constfixed, sf64Matrix<D> &sharedFixed,
                              bool mode) {
    sf64Matrix<D> temp = sharedFixed;

    if (partyIdx == 0)
      for (i64 i = 0; i < sharedFixed.rows(); i++)
        for (i64 j = 0; j < sharedFixed.cols(); j++)
          temp[0](i, j) = sharedFixed[0](i, j) - constfixed.mValue;
    else if (partyIdx == 1)
      for (i64 i = 0; i < sharedFixed.rows(); i++)
        for (i64 j = 0; j < sharedFixed.cols(); j++)
          temp[1](i, j) = sharedFixed[1](i, j) - constfixed.mValue;
    if (mode != true) {
      temp[0] = -temp[0];
      temp[1] = -temp[1];
    }
    return temp;
  }

  si64 MPC_Sub_Const(i64 constInt, si64 &sharedInt, bool mode);

  si64Matrix MPC_Sub_Const(i64 constInt, si64Matrix &sharedIntMatrix,
                           bool mode);

  template <Decimal D>
  sf64<D> MPC_Mul(std::vector<sf64<D>> sharedFixedInt, sf64<D> &prod) {
    prod = sharedFixedInt[0];
    for (u64 i = 1; i < sharedFixedInt.size(); ++i)
      eval.asyncMul(runtime, prod, sharedFixedInt[i], prod).get();
    return prod;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Mul(std::vector<sf64Matrix<D>> sharedFixedInt) {
    sf64Matrix<D> prod;
    prod = sharedFixedInt[0];
    for (u64 i = 1; i < sharedFixedInt.size(); ++i)
      eval.asyncMul(runtime, prod, sharedFixedInt[i], prod).get();
    return prod;
  }

  si64Matrix MPC_Mul(std::vector<si64Matrix> sharedInt);

  template <Decimal D>
  sf64Matrix<D> MPC_Dot_Mul(const sf64Matrix<D> &A, const sf64Matrix<D> &B) {
    if (A.cols() != B.cols() || A.rows() != B.rows())
      throw std::runtime_error(LOCATION);

    sf64Matrix<D> ret(A.rows(), A.cols());
    eval.asyncDotMul(runtime, A, B, ret).get();
    return ret;
  }

  si64Matrix MPC_Dot_Mul(const si64Matrix &A, const si64Matrix &B);

  template <Decimal D>
  sf64<D> MPC_Mul_Const(const f64<D> &constFixed, const sf64<D> &sharedFixed) {
    sf64<D> ret;
    eval.asyncConstFixedMul(runtime, constFixed, sharedFixed, ret).get();
    return ret;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Mul_Const(const f64<D> &constFixed,
                              const sf64Matrix<D> &sharedFixed) {
    sf64Matrix<D> ret(sharedFixed.rows(), sharedFixed.cols());
    eval.asyncConstFixedMul(runtime, constFixed, sharedFixed, ret).get();
    return ret;
  }

  si64 MPC_Mul_Const(const i64 &constInt, const si64 &sharedInt);

  si64Matrix MPC_Mul_Const(const i64 &constInt,
                           const si64Matrix &sharedIntMatrix);

  template <Decimal D> sf64Matrix<D> MPC_Abs(const sf64Matrix<D> &Y) {
    if (mAbs.mThresholds.size() == 0) {
      mAbs.mThresholds.resize(1);
      mAbs.mThresholds[0] = 0;
      mAbs.mCoefficients.resize(2);
      mAbs.mCoefficients[0].resize(2);
      mAbs.mCoefficients[0][0] = 0;
      mAbs.mCoefficients[0][1] = -1;
      mAbs.mCoefficients[1].resize(2);
      mAbs.mCoefficients[1][0] = 0;
      mAbs.mCoefficients[1][1] = 1;
    }

    sf64Matrix<D> out(Y.rows(), Y.cols());
    mAbs.eval<D>(runtime.noDependencies(), Y, out, eval);
    return out;
  }

  template <Decimal D> sf64Matrix<D> MPC_QuoDertermine(const sf64Matrix<D> &Y) {
    if (mQuoDertermine.mThresholds.size() == 0) {
      mQuoDertermine.mThresholds.resize(1);
      mQuoDertermine.mThresholds[0] = 0;
      mQuoDertermine.mCoefficients.resize(2);
      mQuoDertermine.mCoefficients[0].resize(1);
      mQuoDertermine.mCoefficients[0][0] = -1;
      mQuoDertermine.mCoefficients[1].resize(1);
      mQuoDertermine.mCoefficients[1][0] = 1;
    }

    sf64Matrix<D> out(Y.rows(), Y.cols());
    mQuoDertermine.eval<D>(runtime.noDependencies(), Y, out, eval);
    return out;
  }

  template <Decimal D> sf64Matrix<D> MPC_DReLu(const sf64Matrix<D> &Y) {
    if (mdivision.mThresholds.size() == 0) {
      mdivision.mThresholds.resize(1);
      mdivision.mThresholds[0] = 0;
      mdivision.mCoefficients.resize(2);
      mdivision.mCoefficients[0].resize(2);
      mdivision.mCoefficients[0][0] = 0;
      mdivision.mCoefficients[0][1] = 0;
      mdivision.mCoefficients[1].resize(2);
      mdivision.mCoefficients[1][0] = 1;
      mdivision.mCoefficients[1][1] = 0;
    }

    sf64Matrix<D> out(Y.rows(), Y.cols());
    mdivision.eval<D>(runtime.noDependencies(), Y, out, eval);
    return out;
  }
  //>0.5
  template <Decimal D> eMatrix<i64> MPC_Pow(const sf64Matrix<D> &Y) {
    sf64Matrix<D> Y_temp(Y.rows(), Y.cols());
    sf64Matrix<D> drelu_result(Y.rows(), Y.cols());
    eMatrix<i64> Alpha_matrix(Y.rows(), Y.cols());
    for (int k = 0; k < Y.rows(); k++) {
      for (int j = 0; j < Y.cols(); j++) {
        Alpha_matrix(k, j) = 0;
      }
    }
    // std::cout << "Alpha matrix initial: " << Alpha_matrix << std::endl;
    eMatrix<double> drelu_result_temp(Y.rows(), Y.cols());
    eMatrix<u64> rank_matrix(Y.rows(), Y.cols());
    for (int i = 5; i >= 0; i--) {
      // std::cout << "The " << i << "th time iteration" << std::endl;
      // here we calculate the rank in each iteration. <<,>>can only use in
      // integer type
      u64 round_bound = 1 << i;
      // u64 rank <<=  D;
      // if (partyIdx == 0) {
      //   std::cout << "round_bound: " << round_bound << std::endl;
      // }
      for (int k = 0; k < Y.rows(); k++) {
        for (int j = 0; j < Y.cols(); j++) {
          rank_matrix(k, j) = Alpha_matrix(k, j) + round_bound;
        }
      }

      f64Matrix<D> rank_temp(Y.rows(), Y.cols());
      for (int k = 0; k < Y.rows(); k++) {
        for (int j = 0; j < Y.cols(); j++) {
          rank_temp(k, j) = 1ULL << rank_matrix(k, j);
        }
      }
      sf64Matrix<D> sfrank_temp(Y.rows(), Y.cols());
      if (partyIdx == 0) { // key point here is we can use partyIdx == 0 to
                           // present the input secret belongs to whom.
        enc.localFixedMatrix(runtime, rank_temp, sfrank_temp).get();
      } else {
        enc.remoteFixedMatrix(runtime, sfrank_temp).get();
      }
      Y_temp = Y - sfrank_temp;

      /*
        what is mShares[0] means to?
        what will happen if we use constant * mshares[0]?
      */
      // }
      sleep(1);
      // std::cout << "party: " << partyIdx << "Y_temp result by (x -
      // rank_matrix)"
      //           << revealAll(Y_temp).format(HeavyFmt) << std::endl;
      // here we use MPC_DReLu to calculate the signal of (x-rank) and get 1 or
      // 0.

      drelu_result = MPC_DReLu(Y_temp);
      std::cout << i << "th Drelu result for Y_temp"
                << revealAll(drelu_result).format(HeavyFmt) << std::endl;

      drelu_result_temp = revealAll(
          drelu_result); // ematrix should mutiply rank to get rank matrix.
      // rank_matrix += rank * drelu_result_temp;//check if this is working!!!
      for (u64 i = 0; i < Alpha_matrix.size(); ++i)
        Alpha_matrix(i) += round_bound * static_cast<u64>(drelu_result_temp(i));

    } // 5 rounds iteration.
    for (u64 i = 0; i < Alpha_matrix.rows(); i++)
      for (u64 j = 0; j < Alpha_matrix.cols(); j++) {
        Alpha_matrix(i, j) = Alpha_matrix(i, j) + 1;
      }
    if (partyIdx == 0)
      if (partyIdx == 0) {
        std::cout << "final partyIdx: " << partyIdx << std::endl;
        std::cout << "final Alpha_matrix: " << Alpha_matrix << std::endl;
      }

    return Alpha_matrix;
  }

  //<0.5
  template <Decimal D> eMatrix<i64> MPC_Pow2(const sf64Matrix<D> &Y) {
    sf64Matrix<D> Y_temp(Y.rows(), Y.cols());
    sf64Matrix<D> drelu_result(Y.rows(), Y.cols());
    eMatrix<i64> Alpha_matrix(Y.rows(), Y.cols());
    for (int k = 0; k < Y.rows(); k++) {
      for (int j = 0; j < Y.cols(); j++) {
        Alpha_matrix(k, j) = 0;
      }
    }
    // std::cout << "Alpha matrix initial: " << Alpha_matrix << std::endl;
    eMatrix<double> drelu_result_temp(Y.rows(), Y.cols());
    eMatrix<i64> rank_matrix(Y.rows(), Y.cols());
    for (int i = 5; i >= 0; i--) {
      // std::cout << "The " << i << "th time iteration" << std::endl;
      // here we calculate the rank in each iteration. <<,>>can only use in
      // integer type
      u64 round_bound = 1 << i;
      // u64 rank <<=  D;
      // if (partyIdx == 0) {
      //   std::cout << "round_bound: " << round_bound << std::endl;
      // }
      for (int k = 0; k < Y.rows(); k++) {
        for (int j = 0; j < Y.cols(); j++) {
          rank_matrix(k, j) = Alpha_matrix(k, j) + round_bound;
        }
      }

      f64Matrix<D> rank_temp(Y.rows(), Y.cols());
      for (int k = 0; k < Y.rows(); k++) {
        for (int j = 0; j < Y.cols(); j++) {
          rank_temp(k, j) = pow(2, rank_matrix(k, j) * (-1));
        }
      }
      if (partyIdx == 0) {
        std::cout << partyIdx << " rank_temp: " << rank_matrix << std::endl;
      }
      sf64Matrix<D> sfrank_temp(Y.rows(), Y.cols());
      if (partyIdx == 0) { // key point here is we can use partyIdx == 0 to
                           // present the input secret belongs to whom.
        enc.localFixedMatrix(runtime, rank_temp, sfrank_temp).get();
      } else {
        enc.remoteFixedMatrix(runtime, sfrank_temp).get();
      }
      // Y_temp = Y - sfrank_temp;
      Y_temp = sfrank_temp - Y;

      /*
        what is mShares[0] means to?
        what will happen if we use constant * mshares[0]?
      */
      // }
      sleep(1);
      // std::cout << "party: " << partyIdx << "Y_temp result by (x -
      // rank_matrix)"
      //           << revealAll(Y_temp).format(HeavyFmt) << std::endl;
      // here we use MPC_DReLu to calculate the signal of (x-rank) and get 1 or
      // 0.

      drelu_result = MPC_DReLu(Y_temp);
      std::cout << i << "th Drelu result for Y_temp"
                << revealAll(drelu_result).format(HeavyFmt) << std::endl;

      drelu_result_temp = revealAll(
          drelu_result); // ematrix should mutiply rank to get rank matrix.
      // rank_matrix += rank * drelu_result_temp;//check if this is working!!!
      for (u64 i = 0; i < Alpha_matrix.size(); ++i)
        Alpha_matrix(i) += round_bound * static_cast<u64>(drelu_result_temp(i));

    } // 5 rounds iteration.
    for (u64 i = 0; i < Alpha_matrix.rows(); i++)
      for (u64 j = 0; j < Alpha_matrix.cols(); j++) {
        Alpha_matrix(i, j) = Alpha_matrix(i, j) * (-1);
      }
    if (partyIdx == 0)
      if (partyIdx == 0) {
        std::cout << "final partyIdx: " << partyIdx << std::endl;
        std::cout << "final Alpha_matrix: " << Alpha_matrix << std::endl;
      }

    return Alpha_matrix;
  }

  template <Decimal D>
  vector<sf64<D>> MPC_sfmatrixTosfvec(const sf64Matrix<D> &X) {
    vector<sf64<D>> dest(X.size());
    int count = 0;
    for (int k = 0; k < X.rows(); k++) {
      for (int j = 0; j < X.cols(); j++) {
        dest[count][0] = X[0](k, j);
        dest[count][1] = X[1](k, j);
        count++;
      }
    }
    return dest;
  }

  template <Decimal D>
  sf64Matrix<D> MPC_sfvecTosfmatrix(const vector<sf64<D>> &X, int rows,
                                    int cols) {
    assert(rows * cols == X.size() &&
           "Input vector size should be consistent of output matrix!");
    sf64Matrix<D> dest(rows, cols);
    int count = 0;
    for (int k = 0; k < dest.rows(); k++) {
      for (int j = 0; j < dest.cols(); j++) {
        dest[0](k, j) = X[count][0];
        dest[1](k, j) = X[count][1];
        count++;
      }
    }
    return dest;
  }

  template <Decimal D>
  void MPC_Dotproduct(const sf64Matrix<D> &A, const sf64Matrix<D> &B,
                      sf64Matrix<D> &C, u64 shift = 0) {
    assert(A.cols() == B.cols() && A.rows() == B.rows() &&
           "Size of A and B should be completely consistent.");

    eval.asyncDotMul(runtime, A, B, C).get();
  }

  template <Decimal D>
  void MPC_Dotproduct(const sf64Matrix<D> &A, const sf64Matrix<D> &B,
                      sf64Matrix<D> &C, eMatrix<u64> shift) {
    assert(A.cols() == B.cols() && A.rows() == B.rows() &&
           "Size of A and B should be completely consistent.");

    eval.asyncDotMul(runtime, A, B, C, D);
  }

  template <Decimal D>
  sf64Matrix<D> MPC_Div(const sf64Matrix<D> &A, const sf64Matrix<D> &B) {
    if (A.cols() != B.cols() || A.rows() != B.rows())
      throw std::runtime_error(LOCATION);
    sf64Matrix<D> ret(A.rows(), B.cols());

    sf64Matrix<D> denominator_sign(B.rows(), B.cols());
    sf64Matrix<D> denominator(B.rows(), B.cols());
    denominator_sign = MPC_QuoDertermine(B);
    denominator = MPC_Abs(B);
    sf64Matrix<D> numerator_sign(A.rows(), A.cols());
    sf64Matrix<D> numerator(A.rows(), A.cols());
    numerator_sign = MPC_QuoDertermine(A);
    numerator = MPC_Abs(A);

    sf64Matrix<D> quotient_sign(B.rows(), B.cols());
    MPC_Dotproduct(denominator_sign, numerator_sign, quotient_sign);

    //.................................................................................
    // judge whether denominator >=0.5 or <0.5
    f64Matrix<D> zeropointfive(B.rows(), B.cols());
    for (u64 i = 0; i < B.rows(); i++) {
      zeropointfive(i, 0) = 0.5;
    }
    sf64Matrix<D> sfzeropointfive(B.rows(), B.cols());
    ;
    if (partyIdx == 0) {
      enc.localFixedMatrix(runtime, zeropointfive, sfzeropointfive).get();
    } else {
      enc.remoteFixedMatrix(runtime, sfzeropointfive).get();
    }

    sf64Matrix<D> diff_temp(B.rows(), B.cols());
    diff_temp = denominator - sfzeropointfive;

    sf64Matrix<D> drelu_result(B.rows(), B.cols());
    drelu_result = MPC_DReLu(diff_temp);
    eMatrix<double> drelu_result_temp(B.rows(), B.cols());
    drelu_result_temp = revealAll(drelu_result);

    vector<u64> lessIdx;
    vector<u64> moreIdx;

    for (u64 i = 0; i < drelu_result.rows(); i++) {
      //<0.5
      if (drelu_result_temp(i, 0) == 0) {
        lessIdx.push_back(i);
      } else {
        moreIdx.push_back(i);
      }
    }

    // std::cout <<"P"<<partyIdx<< " : denominator(lessIdx[i],0): " <<
    // denominator[u64(0)].format(HeavyFmt) << std::endl;

    sf64Matrix<D> denominatorLess(lessIdx.size(), B.cols());
    sf64Matrix<D> denominatorMore(moreIdx.size(), B.cols());
    f64<D> temp;
    for (u64 i = 0; i < lessIdx.size(); i++) {
      std::cout << "lessIdx: " << lessIdx[i] << std::endl;
      denominatorLess[0](i, 0) = denominator[0](lessIdx[i], 0);
      denominatorLess[1](i, 0) = denominator[1](lessIdx[i], 0);

      //   revealAll(denominator(lessIdx[i],0), temp);
    }
    for (u64 i = 0; i < moreIdx.size(); i++) {
      denominatorMore[0](i, 0) = denominator[0](moreIdx[i], 0);
      denominatorMore[1](i, 0) = denominator[1](moreIdx[i], 0);

      std::cout << "moreIdx: " << moreIdx[i] << std::endl;
      // std::cout << "denominator(MoreIdx[i],0)): " <<
      // revealAll(denominator(moreIdx[i],0)).format(HeavyFmt) << std::endl;
    }
    std::cout << "moreIdx.size(): " << moreIdx.size() << std::endl;

    std::cout << "denominator: " << revealAll(denominator).format(HeavyFmt)
              << std::endl;

    std::cout << "denominatorLess: "
              << revealAll(denominatorLess).format(HeavyFmt) << std::endl;
    std::cout << "denominatorMore: "
              << revealAll(denominatorMore).format(HeavyFmt) << std::endl;

    eMatrix<i64> rankLess(lessIdx.size(), B.cols());
    eMatrix<i64> rankMore(moreIdx.size(), B.cols());
    rankLess = MPC_Pow2(denominatorLess);
    rankMore = MPC_Pow(denominatorMore);
    eMatrix<i64> rank(B.rows(), B.cols());
    for (u64 i = 0; i < lessIdx.size(); i++) {
      rank(lessIdx[i], 0) = rankLess(i, 0);
    }
    for (u64 i = 0; i < moreIdx.size(); i++) {
      rank(moreIdx[i], 0) = rankMore(i, 0);
    }
    //.................................................................................

    /*because of the limitation of PIECEWISE, we have to input n rows and 1
     * cols*/
    // w0 = 2.9142-2b and 1 Note:2.9142 and 1 has been truncate by rank+1;
    // eMatrix<i64> rank = MPC_Pow(denominator);
    eMatrix<i64> precision(B.rows(), B.cols());
    eMatrix<i64> double_precision(B.rows(), B.cols());
    // eMatrix<double> constant_two(B.rows(),B.cols());
    f64Matrix<D> twopotnine(B.rows(), B.cols());
    f64Matrix<D> constant_one(B.rows(), B.cols());
    f64Matrix<D> normalization(B.rows(), B.cols());

    // The divisor is directly normalized,
    // so no additional truncation of the precision bit is required
    // and matrix dot multiplication can be performed directly without
    // recirculation
    for (int k = 0; k < B.rows(); k++) {
      for (int j = 0; j < B.cols(); j++) {
        twopotnine(k, j) = 2.9142;
        constant_one(k, j) = 1;
        precision(k, j) = rank(k, j);
        double_precision(k, j) = 2 * precision(k, j);
        // normalization
        normalization(k, j) = pow(2, (i64)(rank(k, j)) * (-1));
      }
    }

    sf64Matrix<D> sftwopotnine(B.rows(), B.cols());
    sf64Matrix<D> sfconstant_one(B.rows(), B.cols());
    sf64Matrix<D> sfnormalization(B.rows(), B.cols());
    // get shares
    if (partyIdx == 0) {
      enc.localFixedMatrix(runtime, twopotnine, sftwopotnine).get();
      enc.localFixedMatrix(runtime, constant_one, sfconstant_one).get();
      enc.localFixedMatrix(runtime, normalization, sfnormalization).get();
    } else {
      enc.remoteFixedMatrix(runtime, sftwopotnine).get();
      enc.remoteFixedMatrix(runtime, sfconstant_one).get();
      enc.remoteFixedMatrix(runtime, sfnormalization).get();
    }
    std::cout << "sftwopotnine result: "
              << revealAll(sftwopotnine).format(HeavyFmt) << std::endl;
    std::cout << "sfconstant_one result: "
              << revealAll(sfconstant_one).format(HeavyFmt) << std::endl;
    std::cout << "sfnormalization result: "
              << revealAll(sfnormalization).format(HeavyFmt) << std::endl;
    sf64Matrix<D> temp_twob(B.rows(), B.cols());
    sf64Matrix<D> w0(B.rows(), B.cols());
    i64 constant_two = 2;
    eval.asyncConstMul(constant_two, denominator,
                       temp_twob); // const needn't .get()
    //...............................................................................

    sf64Matrix<D> c(B.rows(), B.cols());
    // truncate D，no additional truncation
    MPC_Dotproduct(denominator, sfnormalization, c, 0);
    std::cout << "c result: " << revealAll(c).format(HeavyFmt) << std::endl;
    sf64Matrix<D> temp_twoc(B.rows(), B.cols());
    eval.asyncConstMul(constant_two, c,
                       temp_twoc); // const needn't .get()
    //...............................................................................
    // The initial approximation of 1/c
    w0 = sftwopotnine - temp_twoc; // here means w0 has been truncate by rank+1;
    std::cout << "1/c w0 result: " << revealAll(w0).format(HeavyFmt)
              << std::endl;
    // The initial approximation of 1/b
    MPC_Dotproduct(w0, sfnormalization, w0, 0);
    std::cout << "1/b w0 result: " << revealAll(w0).format(HeavyFmt)
              << std::endl;

    sf64Matrix<D> epsilon0(B.rows(), B.cols());

    sf64Matrix<D> bw0(B.rows(), B.cols());

    LOG(INFO) << "denominator result: "
              << revealAll(denominator).format(HeavyFmt);

    MPC_Dotproduct(denominator, w0, bw0, 0);
    std::cout << "bw0 result: " << revealAll(bw0).format(HeavyFmt) << std::endl;

    // The initial approximation of a/b
    sf64Matrix<D> aw0(B.rows(), B.cols());
    MPC_Dotproduct(numerator, w0, aw0, 0);
    std::cout << "aw0 result: " << revealAll(aw0).format(HeavyFmt) << std::endl;
    // Initial error
    epsilon0 = sfconstant_one - bw0;
    std::cout << "epsilon0 result: " << revealAll(epsilon0).format(HeavyFmt)
              << std::endl;
    //..................................................................................................................................
    sf64Matrix<D> epsilon0_one(B.rows(), B.cols());
    epsilon0_one = sfconstant_one + epsilon0;
    std::cout << "epsilon0_one result: "
              << revealAll(epsilon0_one).format(HeavyFmt) << std::endl;
    sf64Matrix<D> epsilon_prod(B.rows(), B.cols());
    // Closer and closer to a/b (the problem!! Offside will happen)
    MPC_Dotproduct(epsilon0_one, aw0, epsilon_prod, 0);
    sf64Matrix<D> epsilon_pre(B.rows(), B.cols());
    epsilon_pre = epsilon0;
    // The more iterations, the greater the chance of offside
    for (int i = 0; i < 1; i++) {
      sf64Matrix<D> epsilon_i(B.rows(), B.cols());
      MPC_Dotproduct(epsilon_pre, epsilon_pre, epsilon_i, 0);
      sf64Matrix<D> epsilon_i_one(B.rows(), B.cols());
      epsilon_i_one = sfconstant_one + epsilon_i;
      MPC_Dotproduct(epsilon_prod, epsilon_i_one, epsilon_prod, 0);
      epsilon_pre = epsilon_i;
    }
    //..................................................................................................................................

    // get final result:
    MPC_Dotproduct(epsilon_prod, quotient_sign, ret);
    std::cout << revealAll(ret).format(HeavyFmt) << std::endl;
    return ret;
  }

  template <Decimal D> void MPC_Compare(f64Matrix<D> &m, sbMatrix &sh_res) {
    // Get matrix shape of all party.
    std::vector<std::array<uint64_t, 2>> all_party_shape;
    for (uint64_t i = 0; i < 3; i++) {
      std::array<uint64_t, 2> shape;
      if (partyIdx == i) {
        shape[0] = m.rows();
        shape[1] = m.cols();
        mNext.asyncSendCopy(shape);
        mPrev.asyncSendCopy(shape);
      } else {
        if (partyIdx == (i + 1) % 3)
          mPrev.recv(shape);
        else if (partyIdx == (i + 2) % 3)
          mNext.recv(shape);
        else
          throw std::runtime_error("Message recv logic error.");
      }

      all_party_shape.emplace_back(shape);
    }

    {
      LOG(INFO) << "Dump shape of all party's matrix:";
      for (uint64_t i = 0; i < 3; i++)
        LOG(INFO) << "Party " << i << ": (" << all_party_shape[i][0] << ", "
                  << all_party_shape[i][1] << ")";
      LOG(INFO) << "Dump finish.";
    }

    // The compare is a binary operator, so only two party will provide value
    // for compare, find out which party don't provide value.
    int skip_index = -1;
    for (uint64_t i = 0; i < 3; i++) {
      if (all_party_shape[i][0] == 0 && all_party_shape[i][1] == 0) {
        if (skip_index != -1) {
          throw std::runtime_error(
              "There are two party that don't provide any value at last, but "
              "compare operator require value from two party.");
        } else {
          skip_index = i;
        }
      }
    }

    if (skip_index == -1)
      throw std::runtime_error(
          "This operator is binary, can only handle value from two party.");

    // Shape of matrix in two party shoubld be the same.
    if (skip_index == 0) {
      all_party_shape[0][0] = all_party_shape[1][0];
      all_party_shape[0][1] = all_party_shape[1][1];
      all_party_shape[1][0] = all_party_shape[2][0];
      all_party_shape[1][1] = all_party_shape[2][1];
    } else if (skip_index != 2) {
      all_party_shape[1][0] = all_party_shape[2][0];
      all_party_shape[1][1] = all_party_shape[2][1];
    }

    if ((all_party_shape[0][0] != all_party_shape[1][0]) ||
        (all_party_shape[0][1] != all_party_shape[1][1]))
      throw std::runtime_error(
          "Shape of matrix in two party must be the same.");

    // Set value to it's negative for some party.
    if (skip_index == 0 || skip_index == 1) {
      if (partyIdx == 2)
        m.mData = m.mData.array() * -1;
    } else {
      if (partyIdx == 1)
        m.mData = m.mData.array() * -1;
    }

    LOG(INFO) << "Party " << (skip_index + 1) % 3 << " and party "
              << (skip_index + 2) % 3 << " provide value for MPC compare.";

    // Create binary shares.
    uint64_t num_elem = all_party_shape[0][0] * all_party_shape[0][1];
    m.resize(num_elem, 1);

    std::vector<sbMatrix> sh_m_vec;
    for (uint64_t i = 0; i < 3; i++) {
      if (static_cast<int>(i) == skip_index)
        continue;
      if (i == partyIdx) {
        sbMatrix sh_m(num_elem, VAL_BITCOUNT);
        auto task = runtime.noDependencies();
        task.get();
        enc.localBinMatrix(task, m.i64Cast(), sh_m).get();
        sh_m_vec.emplace_back(sh_m);
      } else {
        sbMatrix sh_m(num_elem, VAL_BITCOUNT);
        auto task = runtime.noDependencies();
        task.get();
        enc.remoteBinMatrix(task, sh_m).get();
        sh_m_vec.emplace_back(sh_m);
      }
    }

    LOG(INFO) << "Create binary share for value from two party finish.";

    // Build then run MSB circuit.
    KoggeStoneLibrary lib;
    uint64_t size = 64;
    BetaCircuit *cir = lib.int_int_add_msb(size);
    cir->levelByAndDepth();

    sh_res.resize(m.size(), 1);
    std::vector<const sbMatrix *> input = {&sh_m_vec[0], &sh_m_vec[1]};
    std::vector<sbMatrix *> output = {&sh_res};
    auto task = runtime.noDependencies();
    task = binEval.asyncEvaluate(task, cir, gen, input, output);
    task.get();

    // Recover original value.
    if (skip_index == 0 || skip_index == 1) {
      if (partyIdx == 2)
        m.mData = m.mData.array() * -1;
    } else {
      if (partyIdx == 1)
        m.mData = m.mData.array() * -1;
    }

    LOG(INFO) << "Finish evalute MSB circuit.";
  }

  void MPC_Compare(i64Matrix &m, sbMatrix &sh_res);

  void MPC_Compare(sbMatrix &sh_res);
};
#endif
} // namespace primihub
