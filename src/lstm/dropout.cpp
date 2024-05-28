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

#include "networkio.h"
#include "dropout.h"
#include "networkscratch.h"
#include "serialis.h"
#include <random> // Include the C++ standard random library

namespace tesseract {

DropoutLayer::DropoutLayer(const std::string &name, int ni, float dropout_rate)
    : Network(NT_DROPOUT, name, ni, ni), dropout_rate_(dropout_rate), rng_(std::random_device()()) {}

bool DropoutLayer::Serialize(TFile *fp) const {
  return Network::Serialize(fp) && fp->Serialize(&dropout_rate_);
}

bool DropoutLayer::DeSerialize(TFile *fp) {
  return Network::DeSerialize(fp) && fp->DeSerialize(&dropout_rate_);
}

void DropoutLayer::GenerateDropoutMask(const NetworkIO &input, NetworkIO *dropout_mask) {
  dropout_mask->Resize(input, input.NumFeatures());
  int width = input.Width();
  int num_features = input.NumFeatures();
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int t = 0; t < width; ++t) {
    float *mask_line = dropout_mask->f(t);
    for (int i = 0; i < num_features; ++i) {
      mask_line[i] = dist(rng_) < dropout_rate_ ? 0.0f : 1.0f;
    }
  }
}

void DropoutLayer::ApplyDropout(NetworkIO *output) {
  int width = output->Width();
  int num_features = output->NumFeatures();
  float scale = 1.0f / (1.0f - dropout_rate_);
  for (int t = 0; t < width; ++t) {
    float *line = output->f(t);
    for (int i = 0; i < num_features; ++i) {
      if (line[i] != 0.0f) {
        line[i] *= scale;
      }
    }
  }
}

void DropoutLayer::Forward(bool debug, const NetworkIO &input, const TransposedArray *input_transpose,
                           NetworkScratch *scratch, NetworkIO *output) {
  output->Resize(input, input.NumFeatures());
  if (dropout_rate_ > 0.0f) {
    GenerateDropoutMask(input, &dropout_mask_);
    for (int t = 0; t < input.Width(); ++t) {
      for (int d = 0; d < input.NumFeatures(); ++d) {
        output->f(t)[d] = input.f(t)[d] * dropout_mask_.f(t)[d];
      }
    }
    ApplyDropout(output);
  } else {
    output->CopyAll(input);
  }
}

bool DropoutLayer::Backward(bool debug, const NetworkIO &fwd_deltas, NetworkScratch *scratch,
                            NetworkIO *back_deltas) {
  back_deltas->Resize(fwd_deltas, fwd_deltas.NumFeatures());
  if (dropout_rate_ > 0.0f) {
    for (int t = 0; t < fwd_deltas.Width(); ++t) {
      for (int d = 0; d < fwd_deltas.NumFeatures(); ++d) {
        back_deltas->f(t)[d] = fwd_deltas.f(t)[d] * dropout_mask_.f(t)[d];
      }
    }
  } else {
    back_deltas->CopyAll(fwd_deltas);
  }
  return true;
}

// Provides debug output on the weights.
void DropoutLayer::DebugWeights() {
  tprintf("Dropout layer with rate %f\n", dropout_rate_);
}

}  // namespace tesseract
