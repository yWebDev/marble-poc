import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { TfCoreDemoComponent } from './views/tf-core-demo/tf-core-demo.component';
import { HomeComponent } from './views/home/home.component';
import { TfPlotDemoComponent } from './views/tf-plot-demo/tf-plot-demo.component';
import { TfModelTrainComponent } from './views/tf-model-train/tf-model-train.component';

@NgModule({
  declarations: [
    AppComponent,
    TfCoreDemoComponent,
    HomeComponent,
    TfPlotDemoComponent,
    TfModelTrainComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
