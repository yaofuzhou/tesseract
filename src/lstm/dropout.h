///////////////////////////////////////////////////////////////////////
// File:        dropout.h
// Description: Dropout layer for neural network.
// Author:      Yaofu Zhou
//
// (C) Copyright 2024, MORE Health Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#ifndef TESSERACT_LSTM_DROPOUT_H_
#define TESSERACT_LSTM_DROPOUT_H_

#include "network.h"
#include "networkio.h"
#include <random> // Include the C++ standard random library

namespace tesseract {

class DropoutLayer : public Network {
 public:
  TESS_API DropoutLayer(const std::string &name, int ni, float dropout_rate);
  ~DropoutLayer() override = default;

  std::string spec() const override {
    return "Dr" + std::to_string(dropout_rate_);
  }

  bool Serialize(TFile *fp) const override;
  bool DeSerialize(TFile *fp) override;

  void Forward(bool debug, const NetworkIO &input, const TransposedArray *input_transpose,
               NetworkScratch *scratch, NetworkIO *output) override;
  bool Backward(bool debug, const NetworkIO &fwd_deltas, NetworkScratch *scratch,
                NetworkIO *back_deltas) override;
  void DebugWeights() override;

 private:
  void GenerateDropoutMask(const NetworkIO &input, NetworkIO *dropout_mask);
  void ApplyDropout(NetworkIO *output);

  float dropout_rate_;
  NetworkIO dropout_mask_;
  std::mt19937 rng_; // Mersenne Twister random number generator
};

}  // namespace tesseract

#endif  // TESSERACT_LSTM_DROPOUT_H_
