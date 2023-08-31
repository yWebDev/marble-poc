import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfCoreDemoComponent } from './tf-core-demo.component';

describe('TfCoreDemoComponent', () => {
  let component: TfCoreDemoComponent;
  let fixture: ComponentFixture<TfCoreDemoComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [TfCoreDemoComponent]
    });
    fixture = TestBed.createComponent(TfCoreDemoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
