import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { BinaryClassificationComponent } from './views/binary-classification/binary-classification.component';
import { HomeComponent } from './views/home/home.component';
import { MultiClassificationComponent } from './views/multi-classification/multi-classification.component';
import { TfCoreDemoComponent } from './views/tf-core-demo/tf-core-demo.component';
import { TfModelTrainComponent } from './views/tf-model-train/tf-model-train.component';
import { TfPlotDemoComponent } from './views/tf-plot-demo/tf-plot-demo.component';

const routes: Routes = [
  {
    path: '',
    component: HomeComponent
  },
  {
    path: 'tf-core-demo',
    component: TfCoreDemoComponent
  },
  {
    path: 'tf-plot-demo',
    component: TfPlotDemoComponent
  },
  {
    path: 'tf-model-train',
    component: TfModelTrainComponent
  },
  {
    path: 'binary-classification',
    component: BinaryClassificationComponent
  },
  {
    path: 'multi-classification',
    component: MultiClassificationComponent
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {
}
