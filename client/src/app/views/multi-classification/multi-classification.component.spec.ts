import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MultiClassificationComponent } from './multi-classification.component';

describe('MultiClassificationComponent', () => {
  let component: MultiClassificationComponent;
  let fixture: ComponentFixture<MultiClassificationComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [MultiClassificationComponent]
    });
    fixture = TestBed.createComponent(MultiClassificationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
