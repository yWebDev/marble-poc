import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { TfCoreDemoComponent } from './views/tf-core-demo/tf-core-demo.component';
import { HomeComponent } from './views/home/home.component';
import { TfPlotDemoComponent } from './views/tf-plot-demo/tf-plot-demo.component';

@NgModule({
  declarations: [
    AppComponent,
    TfCoreDemoComponent,
    HomeComponent,
    TfPlotDemoComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
