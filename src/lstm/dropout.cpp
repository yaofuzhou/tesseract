///////////////////////////////////////////////////////////////////////
// File:        dropout.cpp
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

#ifdef HAVE_CONFIG_H
#  include "config_auto.h"
#endif

#include "dropout.h"
#include "networkscratch.h"
#include "serialis.h"

namespace tesseract {

DropoutLayer::DropoutLayer(const std::string &name, int ni, float dropout_rate)
    : Network(NT_DROPOUT, name, ni, ni), dropout_rate_(dropout_rate) {}

bool DropoutLayer::Serialize(TFile *fp) const {
  return Network::Serialize(fp) && fp->Serialize(&dropout_rate_);
}

bool DropoutLayer::DeSerialize(TFile *fp) {
  return Network::DeSerialize(fp) && fp->DeSerialize(&dropout_rate_);
}

void DropoutLayer::Forward(bool debug, const NetworkIO &input, const TransposedArray *input_transpose,
                           NetworkScratch *scratch, NetworkIO *output) {
  output->Resize(input, input.NumFeatures());
  if (dropout_rate_ > 0.0f) {
    NetworkIO dropout_mask;
    GenerateDropoutMask(input, &dropout_mask);
    for (int t = 0; t < input.Width(); ++t) {
      for (int d = 0; d < input.NumFeatures(); ++d) {
        output->f(t)[d] = input.f(t)[d] * dropout_mask.f(t)[d] / (1.0f - dropout_rate_);
      }
    }
  } else {
    output->CopyAll(input);
  }
}

void DropoutLayer::GenerateDropoutMask(const NetworkIO &input, NetworkIO *dropout_mask) {
  dropout_mask->Resize(input, input.NumFeatures());
  for (int t = 0; t < input.Width(); ++t) {
    for (int d = 0; d < input.NumFeatures(); ++d) {
      dropout_mask->f(t)[d] = (rand() / (RAND_MAX + 1.0)) > dropout_rate_ ? 1 : 0;
    }
  }
}

bool DropoutLayer::Backward(bool debug, const NetworkIO &fwd_deltas, NetworkScratch *scratch,
                            NetworkIO *back_deltas) {
  back_deltas->Resize(fwd_deltas, fwd_deltas.NumFeatures());
  NetworkIO dropout_mask;
  GenerateDropoutMask(fwd_deltas, &dropout_mask);
  for (int t = 0; t < fwd_deltas.Width(); ++t) {
    for (int d = 0; d < fwd_deltas.NumFeatures(); ++d) {
      back_deltas->f(t)[d] = fwd_deltas.f(t)[d] * dropout_mask.f(t)[d] / (1.0f - dropout_rate_);
    }
  }
  return true;
}

// Provides debug output on the weights.
void DropoutLayer::DebugWeights() {
  tprintf("Dropout layer with rate %f\n", dropout_rate_);
}

}  // namespace tesseract
