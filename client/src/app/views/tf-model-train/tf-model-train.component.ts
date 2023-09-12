import { Component, inject, OnInit } from '@angular/core';
import { TfService } from '../../services/tf.service';

@Component({
  selector: 'app-tf-model-train',
  templateUrl: './tf-model-train.component.html',
  styleUrls: ['./tf-model-train.component.scss']
})
export class TfModelTrainComponent implements OnInit {
  private tf = inject(TfService);

  ngOnInit() {
    this.tf.init().then(() => this.run());
  }

  private run(): void {
    const model = this.createModel();
    model.summary();
    this.tf.vis.show.modelSummary({ name: `Model Summary`, tab: `Model` }, model);
    this.tf.vis.show.layer({ name: `Layer 1`, tab: `Model Inspection` }, model.getLayer('', 0));
    this.tf.vis.show.layer({ name: `Layer 2`, tab: `Model Inspection` }, model.getLayer('', 1));
  }

  private createModel() {
    const model = this.tf.core.sequential();

    // Layer 1
    model.add(this.tf.core.layers.dense({
      inputDim: 1,
      units: 1,
      activation: 'linear',
      useBias: true,
    }));

    // Layer 2
    model.add(this.tf.core.layers.dense({
      inputDim: 1,
      units: 3,
      activation: 'linear',
      useBias: false,
    }));

    const optimizer = this.tf.core.train.sgd(0.1);
    model.compile({
      optimizer,
      loss: 'meanSquaredError'
    });

    return model;
  }
}
