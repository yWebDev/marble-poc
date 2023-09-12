import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfModelTrainComponent } from './tf-model-train.component';

describe('TfModelTrainComponent', () => {
  let component: TfModelTrainComponent;
  let fixture: ComponentFixture<TfModelTrainComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [TfModelTrainComponent]
    });
    fixture = TestBed.createComponent(TfModelTrainComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
