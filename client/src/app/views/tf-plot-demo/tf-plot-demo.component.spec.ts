import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfPlotDemoComponent } from './tf-plot-demo.component';

describe('TfPlotDemoComponent', () => {
  let component: TfPlotDemoComponent;
  let fixture: ComponentFixture<TfPlotDemoComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [TfPlotDemoComponent]
    });
    fixture = TestBed.createComponent(TfPlotDemoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
